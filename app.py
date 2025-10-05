# app.py — Gradio front-end that calls test.py IN-PROCESS (Local GPU)
# Folder layout per run (under TEMP_ROOT):
#   input_video/<video_stem>/00000.png ...
#   ref/<video_stem>/ref.png
#   output/<video_stem>/*.png
# Final mp4: TEMP_ROOT/<video_stem>.mp4

import os
import sys
import shutil
import urllib.request
from os import path
import io
from contextlib import redirect_stdout, redirect_stderr

import gradio as gr
from PIL import Image
import cv2
import torch  # used for cuda device set / sync / empty_cache

# ----------------- BASIC INFO -----------------
CHECKPOINT_URL = "https://github.com/yyang181/colormnet/releases/download/v0.1/DINOv2FeatureV6_LocalAtten_s2_154000.pth"
CHECKPOINT_LOCAL = "DINOv2FeatureV6_LocalAtten_s2_154000.pth"

TITLE = "ColorMNet — 视频着色 / Video Colorization (Local GPU)"
DESC = """
**中文**  
上传**黑白视频**与**参考图像**，点击「开始着色 / Start Coloring」。  
此版本在**本地指定 GPU（如 GPU:0）**上运行，并在**同一进程**调用 `test.py` 的入口函数。  
临时工作目录结构：  
- 抽帧：`_colormnet_tmp/input_video/<视频名>/00000.png ...`  
- 参考：`_colormnet_tmp/ref/<视频名>/ref.png`  
- 输出：`_colormnet_tmp/output/<视频名>/*.png`  
- 合成视频：`_colormnet_tmp/<视频名>.mp4`

**English**  
Upload a **B&W video** and a **reference image**, then click “Start Coloring”.  
This app runs **on a local, user-selected GPU (e.g., GPU:0)** and calls `test.py` **in-process**.  
Temp workspace layout:  
- Frames: `_colormnet_tmp/input_video/<stem>/00000.png ...`  
- Reference: `_colormnet_tmp/ref/<stem>/ref.png`  
- Output frames: `_colormnet_tmp/output/<stem>/*.png`  
- Final video: `_colormnet_tmp/<stem>.mp4`
"""

PAPER = """
### 论文 / Paper
**ECCV 2024 — ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization**  

如果你喜欢这个项目，欢迎到 GitHub 点个 ⭐ Star：  
**GitHub**: https://github.com/yyang181/colormnet

**BibTeX 引用 / BibTeX Citation**
```bibtex
@inproceedings{yang2024colormnet,
  author    = {Yixin Yang and Jiangxin Dong and Jinhui Tang and Jinshan Pan},
  title     = {ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization},
  booktitle = ECCV,
  year      = {2024}
}
"""
BADGES_HTML = """
<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
  <a href="https://github.com/yyang181/colormnet" target="_blank" title="Open GitHub Repo">
    <img alt="GitHub Repo"
         src="https://img.shields.io/badge/GitHub-colormnet-181717?logo=github" />
  </a>
  <a href="https://github.com/yyang181/colormnet/stargazers" target="_blank" title="Star on GitHub">
    <img alt="GitHub Repo stars"
         src="https://img.shields.io/github/stars/yyang181/colormnet?style=social" />
  </a>
</div>
"""

# ----------------- REFERENCE FRAME GUIDE (NO CROPPING) -----------------
REF_GUIDE_MD = r"""
## 参考帧制作指南 / Reference Frame Guide

**目的 / Goal**  
为模型提供一张与你的视频关键帧在**姿态、光照、构图**尽量接近的**彩色参考图**，用来指导整段视频的着色风格与主体颜色。

---

### 中文步骤
1. **挑帧**：从视频里挑一帧（或相近角度的照片），尽量与要着色的镜头在**姿态 / 光照 / 场景**一致。  
2. **上色方式**：若你只有黑白参考图、但需要彩色参考，可用 **通义千问·图像编辑（Qwen-Image）**：  
   - 打开：<https://chat.qwen.ai/> → 选择**图像编辑**  
   - 上传你的黑白参考图  
   - 在提示词里输入：  
     **「帮我给这张照片上色，只修改颜色，不要修改内容」**  
   - 可按需多次编辑（如补充「衣服为复古蓝、肤色自然、不要锐化」）  
3. **保存格式**：PNG/JPG 均可；推荐分辨率 ≥ **480px**（短边）。  
4. **文件放置**：本应用会自动放置为 `ref/<视频名>/ref.png`。  

**注意事项（Do/Don’t）**  
- ✅ 主体清晰、颜色干净，不要过曝或强滤镜。  
- ✅ 关键区域（衣服、皮肤、头发、天空等）颜色与目标风格一致。  
- ❌ 不要更改几何结构（如人脸形状/姿态），**只修改颜色**。  
- ❌ 避免文字、贴纸、重度风格化滤镜。

---

### English Steps
1. **Pick a frame** (or a similar photo) that matches the target shot in **pose / lighting / composition**.  
2. **Colorizing if your reference is B&W** — use **Qwen-Image (Image Editing)**:  
   - Open <https://chat.qwen.ai/> → **Image Editing**  
   - Upload your B&W reference  
   - Prompt: **“Help me colorize this photo; only change colors, do not alter the content.”**  
   - Iterate if needed (e.g., “vintage blue jacket, natural skin tone; avoid sharpening”).  
3. **Format**: PNG/JPG; recommended short side ≥ **480px**.  
4. **File placement**: The app will place it as `ref/<video_stem>/ref.png`.

**Do / Don’t**
- ✅ Clean subject and palette; avoid overexposure/harsh filters.  
- ✅ Ensure key regions (clothes/skin/hair/sky) match the intended colors.  
- ❌ Do not change geometry/structure — **colors only**.  
- ❌ Avoid text/stickers/heavy stylization filters.
"""

# ----------------- TEMP WORKDIR -----------------
TEMP_ROOT = path.join(os.getcwd(), "_colormnet_tmp")
INPUT_DIR = "input_video"
REF_DIR = "ref"
OUTPUT_DIR = "output"

def reset_temp_root():
    """每次运行前清空并重建临时工作目录。"""
    if path.isdir(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)
    os.makedirs(TEMP_ROOT, exist_ok=True)
    for sub in (INPUT_DIR, REF_DIR, OUTPUT_DIR):
        os.makedirs(path.join(TEMP_ROOT, sub), exist_ok=True)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

# ----------------- CHECKPOINT (可选) -----------------
def ensure_checkpoint():
    """若 test.py 会在当前目录加载权重，可提前预下载，避免首次拉取超时。"""
    try:
        if not path.exists(CHECKPOINT_LOCAL):
            print(f"[INFO] Downloading checkpoint from: {CHECKPOINT_URL}")
            urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_LOCAL)
            print("[INFO] Checkpoint downloaded:", CHECKPOINT_LOCAL)
    except Exception as e:
        print(f"[WARN] 预下载权重失败（首次推理会再试）: {e}")

# ----------------- VIDEO UTILS -----------------
def video_to_frames_dir(video_path: str, frames_dir: str):
    """
    抽帧到 frames_dir/00000.png ...
    返回: (w, h, fps, n_frames)
    """
    ensure_dir(frames_dir)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video: {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    w = h = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue
        h, w = frame.shape[:2]
        out_path = path.join(frames_dir, f"{idx:05d}.png")
        ok = cv2.imwrite(out_path, frame)
        if not ok:
            raise RuntimeError(f"写入抽帧失败 / Failed to write: {out_path}")
        idx += 1
    cap.release()
    if idx == 0:
        raise RuntimeError("视频无可读帧 / Input video has no readable frames.")
    return w, h, fps, idx

def encode_frames_to_video(frames_dir: str, out_path: str, fps: float):
    frames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".png")])
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")
    first = cv2.imread(path.join(frames_dir, frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame {frames[0]}")
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        img = cv2.imread(path.join(frames_dir, f))
        if img is None:
            continue
        vw.write(img)
    vw.release()

# ----------------- CLI MAPPING -----------------
CONFIG_TO_CLI = {
    "FirstFrameIsNotExemplar": "--FirstFrameIsNotExemplar",  # bool
    "dataset": "--dataset",
    "split": "--split",
    "save_all": "--save_all",                                # bool
    "benchmark": "--benchmark",                              # bool
    "disable_long_term": "--disable_long_term",              # bool
    "max_mid_term_frames": "--max_mid_term_frames",
    "min_mid_term_frames": "--min_mid_term_frames",
    "max_long_term_elements": "--max_long_term_elements",
    "num_prototypes": "--num_prototypes",
    "top_k": "--top_k",
    "mem_every": "--mem_every",
    "deep_update_every": "--deep_update_every",
    "save_scores": "--save_scores",                          # bool
    "flip": "--flip",                                        # bool
    "size": "--size",
    "reverse": "--reverse",                                  # bool
}

def build_args_list_for_test(d16_batch_path: str,
                             out_path: str,
                             ref_root: str,
                             cfg: dict):
    """
    构造传给 test.run_cli(args_list) 的参数列表。
    - 必传：--d16_batch_path <input_video_root>、--ref_path <ref_root>、--output <output_root>
    """
    args = [
        "--d16_batch_path", d16_batch_path,
        "--ref_path", ref_root,
        "--output", out_path,
    ]
    for k, v in cfg.items():
        if k not in CONFIG_TO_CLI:
            continue
        flag = CONFIG_TO_CLI[k]
        if isinstance(v, bool):
            if v:
                args.append(flag)          # store_true
        elif v is None:
            continue
        else:
            args.extend([flag, str(v)])
    return args

# ----------------- GRADIO HANDLER (Local GPU) -----------------
def gradio_infer(
    debug_shapes,
    gpu_id,                   # <--- 新增：UI 传入的 GPU ID (int)
    bw_video, ref_image,
    first_not_exemplar, dataset, split, save_all, benchmark,
    disable_long_term, max_mid, min_mid, max_long,
    num_proto, top_k, mem_every, deep_update,
    save_scores, flip, size, reverse
):
    # 在任何 CUDA 初始化前，设置 GPU 设备（环境 + torch）
    if gpu_id is None:
        gpu_id = 0
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    except Exception:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 若此时还未触发 CUDA 初始化，下面 set_device 会生效
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 因为我们把可见设备映射成了单卡列表 [gpu_id]->index 0
    except Exception as e:
        print(f"[WARN] set_device failed or CUDA not available: {e}")

    # 1) 基本校验与临时目录
    if bw_video is None:
        return None, "请上传黑白视频 / Please upload a B&W video."
    if ref_image is None:
        return None, "请上传参考图像 / Please upload a reference image."
    reset_temp_root()

    # 2) 解析视频源路径 & 目标 <video_stem>
    if isinstance(bw_video, dict) and "name" in bw_video:
        src_video_path = bw_video["name"]
    elif isinstance(bw_video, str):
        src_video_path = bw_video
    else:
        return None, "无法读取视频输入 / Failed to read video input."

    video_stem = path.splitext(path.basename(src_video_path))[0]

    # 3) 生成临时路径
    input_root = path.join(TEMP_ROOT, INPUT_DIR)     # _colormnet_tmp/input_video
    ref_root   = path.join(TEMP_ROOT, REF_DIR)       # _colormnet_tmp/ref
    output_root= path.join(TEMP_ROOT, OUTPUT_DIR)    # _colormnet_tmp/output
    input_frames_dir = path.join(input_root, video_stem)
    ref_dir = path.join(ref_root, video_stem)
    out_frames_dir = path.join(output_root, video_stem)
    for d in (input_root, ref_root, output_root, input_frames_dir, ref_dir, out_frames_dir):
        ensure_dir(d)

    # 4) 抽帧 -> input_video/<stem>/
    try:
        _w, _h, fps, _n = video_to_frames_dir(src_video_path, input_frames_dir)
    except Exception as e:
        return None, f"抽帧失败 / Frame extraction failed:\n{e}"

    # 5) 参考帧 -> ref/<stem>/ref.png
    ref_png_path = path.join(ref_dir, "ref.png")
    if isinstance(ref_image, Image.Image):
        try:
            ref_image.save(ref_png_path)
        except Exception as e:
            return None, f"保存参考图像失败 / Failed to save reference image:\n{e}"
    elif isinstance(ref_image, str):
        try:
            shutil.copy2(ref_image, ref_png_path)
        except Exception as e:
            return None, f"复制参考图像失败 / Failed to copy reference image:\n{e}"
    else:
        return None, "无法读取参考图像输入 / Failed to read reference image."

    # 6) 收集 UI 配置
    default_config = {
        "FirstFrameIsNotExemplar": True,
        "dataset": "D16_batch",
        "split": "val",
        "save_all": True,
        "benchmark": False,
        "disable_long_term": False,
        "max_mid_term_frames": 10,
        "min_mid_term_frames": 5,
        "max_long_term_elements": 10000,
        "num_prototypes": 128,
        "top_k": 30,
        "mem_every": 5,
        "deep_update_every": -1,
        "save_scores": False,
        "flip": False,
        "size": -1,
        "reverse": False,
    }
    user_config = {
        "FirstFrameIsNotExemplar": bool(first_not_exemplar) if first_not_exemplar is not None else default_config["FirstFrameIsNotExemplar"],
        "dataset": str(dataset) if dataset else default_config["dataset"],
        "split": str(split) if split else default_config["split"],
        "save_all": bool(save_all) if save_all is not None else default_config["save_all"],
        "benchmark": bool(benchmark) if benchmark is not None else default_config["benchmark"],
        "disable_long_term": bool(disable_long_term) if disable_long_term is not None else default_config["disable_long_term"],
        "max_mid_term_frames": int(max_mid) if max_mid is not None else default_config["max_mid_term_frames"],
        "min_mid_term_frames": int(min_mid) if min_mid is not None else default_config["min_mid_term_frames"],
        "max_long_term_elements": int(max_long) if max_long is not None else default_config["max_long_term_elements"],
        "num_prototypes": int(num_proto) if num_proto is not None else default_config["num_prototypes"],
        "top_k": int(top_k) if top_k is not None else default_config["top_k"],
        "mem_every": int(mem_every) if mem_every is not None else default_config["mem_every"],
        "deep_update_every": int(deep_update) if deep_update is not None else default_config["deep_update_every"],
        "save_scores": bool(save_scores) if save_scores is not None else default_config["save_scores"],
        "flip": bool(flip) if flip is not None else default_config["flip"],
        "size": int(size) if size is not None else default_config["size"],
        "reverse": bool(reverse) if reverse is not None else default_config["reverse"],
    }

    # 7) 预下载权重（可选）
    ensure_checkpoint()

    # 8) 同进程调用 test.py
    try:
        import test_app as test  # 确保 test.py 同目录且提供 run_cli(args_list)
    except Exception as e:
        return None, f"导入 test.py 失败 / Failed to import test.py：\n{e}"

    args_list = build_args_list_for_test(
        d16_batch_path=input_root,   # 指向 input_video 根
        out_path=output_root,        # 指向 output 根（test.py 写 output/<stem>/*.png）
        ref_root=ref_root,           # 指向 ref 根（test.py 读 ref/<stem>/ref.png）
        cfg=user_config
    )

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            entry = getattr(test, "run_cli", None)
            if entry is None or not callable(entry):
                raise RuntimeError("test.py 未提供可调用的 run_cli(args_list) 接口。")
            entry(args_list)
        log = f"GPU_ID={gpu_id} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} \n" \
              f"Args: {' '.join(args_list)}\n\n{buf.getvalue()}"
    except Exception as e:
        log = f"GPU_ID={gpu_id} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} \n" \
              f"Args: {' '.join(args_list)}\n\n{buf.getvalue()}\n\nERROR: {e}"
        return None, log

    # 在合成 mp4 之前：清空 CUDA（防止显存占用）
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 9) 合成 mp4：从 output/<stem>/ 帧合成 -> TEMP_ROOT/<stem>.mp4
    out_frames = path.join(output_root, video_stem)
    if not path.isdir(out_frames):
        return None, f"未找到输出帧目录 / Output frame dir not found：{out_frames}\n\n{log}"
    final_mp4 = path.abspath(path.join(TEMP_ROOT, f"{video_stem}.mp4"))
    try:
        encode_frames_to_video(out_frames, final_mp4, fps=fps)
    except Exception as e:
        return None, f"合成视频失败 / Video mux failed：\n{e}\n\n{log}"

    return final_mp4, f"完成 ✅ / Done ✅\n\n{log}"

# ----------------- UI -----------------
with gr.Blocks() as demo:
    gr.Markdown(f"# {TITLE}")
    gr.HTML(BADGES_HTML)
    gr.Markdown(PAPER)
    gr.Markdown(DESC)

    with gr.Accordion("参考帧制作指南 / Reference Frame Guide", open=False):
        gr.Markdown(REF_GUIDE_MD)

    with gr.Row():
        gpu_id = gr.Number(label="GPU ID (e.g., 0 for cuda:0)", value=0, precision=0)
        debug_shapes = gr.Checkbox(label="调试日志 / Debug Logs（仅用于显示更完整日志 / show verbose logs）", value=False)

    with gr.Row():
        inp_video = gr.Video(label="黑白视频（mp4/webm/avi） / B&W Video", interactive=True)
        inp_ref = gr.Image(label="参考图像（RGB） / Reference Image (RGB)", type="pil")
        gr.Examples(
            label="示例 / Examples",
            examples=[["./example/4.mp4", "./example/4.png"]],
            inputs=[inp_video, inp_ref],
            cache_examples=False,
        )

    with gr.Accordion("高级参数设置 / Advanced Settings（传给 test.py / passed to test.py）", open=False):
        with gr.Row():
            first_not_exemplar = gr.Checkbox(label="FirstFrameIsNotExemplar (--FirstFrameIsNotExemplar)", value=True)
            reverse = gr.Checkbox(label="reverse (--reverse)", value=False)
            dataset = gr.Textbox(label="dataset (--dataset)", value="D16_batch")
            split = gr.Textbox(label="split (--split)", value="val")
            save_all = gr.Checkbox(label="save_all (--save_all)", value=True)
            benchmark = gr.Checkbox(label="benchmark (--benchmark)", value=False)
        with gr.Row():
            disable_long_term = gr.Checkbox(label="disable_long_term (--disable_long_term)", value=False)
            max_mid = gr.Number(label="max_mid_term_frames (--max_mid_term_frames)", value=10, precision=0)
            min_mid = gr.Number(label="min_mid_term_frames (--min_mid_term_frames)", value=5, precision=0)
            max_long = gr.Number(label="max_long_term_elements (--max_long_term_elements)", value=10000, precision=0)
            num_proto = gr.Number(label="num_prototypes (--num_prototypes)", value=128, precision=0)
        with gr.Row():
            top_k = gr.Number(label="top_k (--top_k)", value=30, precision=0)
            mem_every = gr.Number(label="mem_every (--mem_every)", value=5, precision=0)
            deep_update = gr.Number(label="deep_update_every (--deep_update_every)", value=-1, precision=0)
            save_scores = gr.Checkbox(label="save_scores (--save_scores)", value=False)
            flip = gr.Checkbox(label="flip (--flip)", value=False)
            size = gr.Number(label="size (--size)", value=-1, precision=0)

    run_btn = gr.Button("开始着色 / Start Coloring (Local GPU)")
    with gr.Row():
        out_video = gr.Video(label="输出视频（着色结果） / Output (Colorized)", autoplay=True)
        status = gr.Textbox(label="状态 / 日志输出 / Status & Logs", interactive=False, lines=16)

    run_btn.click(
        fn=gradio_infer,
        inputs=[
            debug_shapes,
            gpu_id,
            inp_video, inp_ref,
            first_not_exemplar, dataset, split, save_all, benchmark,
            disable_long_term, max_mid, min_mid, max_long,
            num_proto, top_k, mem_every, deep_update,
            save_scores, flip, size, reverse
        ],
        outputs=[out_video, status]
    )

    gr.HTML("<hr/>")
    gr.HTML(BADGES_HTML)

if __name__ == "__main__":
    try:
        ensure_checkpoint()
    except Exception as e:
        print(f"[WARN] 预下载权重失败（首次推理会再试）: {e}")
    # 允许公网分享
    demo.queue(max_size=32).launch(server_name="0.0.0.0", server_port=7860, share=False)