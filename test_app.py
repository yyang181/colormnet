# test.py — In-process callable version (for ZeroGPU stateless)
# Keep original logic; add build_parser(), run_cli(args_list), and run_inference(args)
# Do NOT initialize CUDA at import-time.

import os
from os import path
from argparse import ArgumentParser
import shutil

# 不在这里做 @spaces.GPU 装饰，避免与 app.py 的 @spaces.GPU 双重调度
# import spaces

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from inference.data.test_datasets import DAVISTestDataset_221128_TransColorization_batch
from inference.data.mask_mapper import MaskMapper
from model.network import ColorMNet
from inference.inference_core import InferenceCore

from progressbar import progressbar

from dataset.range_transform import inv_im_trans, inv_lll2rgb_trans

from skimage import color, io
import cv2

try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')


# ----------------- small utils -----------------
def detach_to_cpu(x):
    return x.detach().cpu()

def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np

def lab2rgb_transform_PIL(mask):
    mask_d = detach_to_cpu(mask)
    mask_d = inv_lll2rgb_trans(mask_d)
    im = tensor_to_np_float(mask_d)

    if len(im.shape) == 3:
        im = im.transpose((1, 2, 0))
    else:
        im = im[:, :, None]

    im = color.lab2rgb(im)

    return im.clip(0, 1)


# ----------------- argparse -----------------
def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--model', default='saves/DINOv2FeatureV6_LocalAtten_s2_154000.pth')
    parser.add_argument('--FirstFrameIsNotExemplar', help='Whether the provided reference frame is exactly the first input frame', action='store_true')

    # dataset setting
    parser.add_argument('--d16_batch_path', default='input', help='Point to folder A/ which contains <video_name>/00000.png etc.')
    parser.add_argument('--ref_path', default='ref', help='Kept for parity; dataset will also read ref.png under each video folder when args provided')
    parser.add_argument('--output', default='result', help='Directory to save results')

    parser.add_argument('--reverse', default=False, action='store_true', help='whether to reverse the frame order')
    parser.add_argument('--allow_resume', action='store_true', 
                        help='skip existing videos that have been colorized')

    # For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
    parser.add_argument('--generic_path')
    parser.add_argument('--dataset', help='D16/D17/Y18/Y19/LV1/LV3/G', default='D16_batch')
    parser.add_argument('--split', help='val/test', default='val')
    parser.add_argument('--save_all', action='store_true', 
                        help='Save all frames. Useful only in YouTubeVOS/long-time video')
    parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')
            
    # Long-term memory options
    parser.add_argument('--disable_long_term', action='store_true')
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                    type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

    # Multi-scale options
    parser.add_argument('--save_scores', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--size', default=-1, type=int, 
                        help='Resize the shorter side to this size. -1 to use original resolution. ')
    return parser


# ----------------- core inference -----------------
def run_inference(args):
    """
    真正的推理流程。必须在 ZeroGPU 的调度上下文里被调用（由 app.py 的 @spaces.GPU 包裹）。
    不要在导入模块时做任何 CUDA 初始化。
    """
    config = vars(args)
    config['enable_long_term'] = not config['disable_long_term']

    if args.output is None:
        args.output = f'.output/{args.dataset}_{args.split}'
        print(f'Output path not provided. Defaulting to {args.output}')

    # ----- Data preparation -----
    is_youtube = args.dataset.startswith('Y')
    is_davis = args.dataset.startswith('D')
    is_lv = args.dataset.startswith('LV')

    if is_youtube or args.save_scores:
        out_path = path.join(args.output, 'Annotations')
    else:
        out_path = args.output

    if args.split != 'val':
        raise NotImplementedError('Only split=val is supported in this script.')

    # 数据集：支持 A/<video>/00000.png ... 且读取 A/<video>/ref.png
    meta_dataset = DAVISTestDataset_221128_TransColorization_batch(
        args.d16_batch_path, imset=args.ref_path, size=args.size, args=args
    )
    palette = None  # 兼容保留

    torch.autograd.set_grad_enabled(False)

    # Set up loader list (video readers)
    meta_loader = meta_dataset.get_datasets()

    # Load checkpoint/model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    network = ColorMNet(config, args.model).to(device).eval()
    if args.model is not None:
        # map_location 不指定，按默认走（ZeroGPU 下会在被调度的设备上加载）
        model_weights = torch.load(args.model, map_location=device)
        network.load_weights(model_weights, init_as_zero_if_needed=True)
    else:
        print('No model loaded.')

    total_process_time = 0.0
    total_frames = 0

    # ----- Start eval over videos -----
    for vid_reader in progressbar(meta_loader, max_value=len(meta_dataset), redirect_stdout=True):
        # 注意：ZeroGPU/Spaces 环境不允许子进程多线程加载，保持 num_workers=0
        loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        vid_name = vid_reader.vid_name
        vid_length = len(loader)

        # LT usage check per original logic
        config['enable_long_term_count_usage'] = (
            config['enable_long_term'] and
            (vid_length
                / (config['max_mid_term_frames'] - config['min_mid_term_frames'])
                * config['num_prototypes'])
            >= config['max_long_term_elements']
        )

        mapper = MaskMapper()
        processor = InferenceCore(network, config=config)
        first_mask_loaded = False

        # skip existing videos 
        if args.allow_resume:
            this_out_path = path.join(out_path, vid_name)
            if path.exists(this_out_path):
                print(f'Skipping {this_out_path} because output already exists.')
                continue

        for ti, data in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=not args.benchmark):
                rgb = data['rgb'].to(device)[0]

                msk = data.get('mask')
                if not config['FirstFrameIsNotExemplar']:
                    msk = msk[:, 1:3, :, :] if msk is not None else None
                    
                info = data['info']
                frame = info['frame'][0]
                shape = info['shape']
                need_resize = info['need_resize'][0]

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                # 第一次必须有 mask
                if not first_mask_loaded:
                    if msk is not None:
                        first_mask_loaded = True
                    else:
                        continue

                if args.flip:
                    rgb = torch.flip(rgb, dims=[-1])
                    msk = torch.flip(msk, dims=[-1]) if msk is not None else None

                # Map possibly non-continuous labels to continuous ones
                if msk is not None:
                    msk = torch.Tensor(msk[0]).to(device)
                    if need_resize:
                        msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                    processor.set_all_labels(list(range(1, 3)))
                    labels = range(1, 3)
                else:
                    labels = None
        
                # Run the model on this frame
                if config['FirstFrameIsNotExemplar']:
                    prob = processor.step_AnyExemplar(
                        rgb,
                        msk[:1, :, :].repeat(3, 1, 1) if msk is not None else None,
                        msk[1:3, :, :] if msk is not None else None,
                        labels,
                        end=(ti == vid_length - 1)
                    )
                else:
                    prob = processor.step(rgb, msk, labels, end=(ti == vid_length - 1))

                # Upsample to original size if needed
                if need_resize:
                    prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:, 0]

                end.record()
                torch.cuda.synchronize()
                total_process_time += (start.elapsed_time(end)/1000)
                total_frames += 1

                if args.flip:
                    prob = torch.flip(prob, dims=[-1])

                if args.save_scores:
                    prob = (prob.detach().cpu().numpy() * 255).astype(np.uint8)

                # Save the mask
                if args.save_all or info['save'][0]:
                    this_out_path = path.join(out_path, vid_name)
                    os.makedirs(this_out_path, exist_ok=True)

                    out_mask_final = lab2rgb_transform_PIL(torch.cat([rgb[:1, :, :], prob], dim=0))
                    out_mask_final = (out_mask_final * 255).astype(np.uint8)

                    out_img = Image.fromarray(out_mask_final)
                    out_img.save(os.path.join(this_out_path, frame[:-4] + '.png'))

    print(f'Total processing time: {total_process_time}')
    print(f'Total processed frames: {total_frames}')
    print(f'FPS: {total_frames / total_process_time}')
    print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')

    # 与原版一致：只在 save_scores=False 且特定数据集/子集时打 zip
    if not args.save_scores:
        if is_youtube:
            print('Making zip for YouTubeVOS...')
            shutil.make_archive(path.join(args.output, path.basename(args.output)), 'zip', args.output, 'Annotations')
        elif is_davis and args.split == 'test':
            print('Making zip for DAVIS test-dev...')
            shutil.make_archive(args.output, 'zip', args.output)


# ----------------- public entrypoints -----------------
def run_cli(args_list=None):
    """
    供 app.py 同进程调用：test.run_cli(args_list)
    """
    parser = build_parser()
    args = parser.parse_args(args_list)
    return run_inference(args)


def main():
    """
    保留命令行可运行：python test.py --d16_batch_path A --output result ...
    注意：若在 Hugging Face Spaces/ZeroGPU 无状态环境下直接 run main()，
    需要由上层（如 app.py 的 @spaces.GPU）提供调度上下文。
    """
    run_cli()


if __name__ == '__main__':
    main()