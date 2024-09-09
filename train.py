import datetime
from os import path
import math
import git

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed

from model.trainer import ColorMNetTrainer
from dataset.vos_dataset import DAVISVidevoDataset

from util.logger import TensorboardLogger
from util.configuration import Configuration

import wandb
import socket

from inference.data.test_datasets import DAVISTestDataset_221128_TransColorization_batch
from inference.data.mask_mapper import MaskMapper
from model.network import ColorMNet
from inference.inference_core import InferenceCore

"""
Initial setup
"""
# Init distributed environment
distributed.init_process_group(backend="nccl")
print(f'CUDA Device count: {torch.cuda.device_count()}')

# Parse command line arguments
raw_config = Configuration()
raw_config.parse()

if raw_config['benchmark']:
    torch.backends.cudnn.benchmark = True

# Get current git info
repo = git.Repo(".")
git_info = str(repo.active_branch)+' '+str(repo.head.commit.hexsha)

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(f'I am rank {local_rank} in this world of size {world_size}!')

network_in_memory = None
stages = raw_config['stages']
stages_to_perform = list(stages)
for si, stage in enumerate(stages_to_perform):

    # Set seed to ensure the same initialization
    torch.manual_seed(14159265)
    np.random.seed(14159265)
    random.seed(14159265)

    # Pick stage specific hyperparameters out
    stage_config = raw_config.get_stage_parameters(stage)
    config = dict(**raw_config.args, **stage_config)
    if config['exp_id'] != 'NULL':
        config['exp_id'] = config['exp_id']+'_s%s'%stages[:si+1]

    config['single_object'] = (stage == '0')

    config['num_gpus'] = world_size
    if config['batch_size']//config['num_gpus']*config['num_gpus'] != config['batch_size']:
        raise ValueError('Batch size must be divisible by the number of GPUs.')
    config['batch_size'] //= config['num_gpus']
    config['num_workers'] //= config['num_gpus']
    print(f'We are assuming {config["num_gpus"]} GPUs.')

    print(f'We are now starting stage {stage}')

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ColorMNet",
        
        notes=socket.gethostname(),
        name=config['exp_id'],
        dir=config['savepath'],
        job_type="training",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "DAVIS_Videvo",
        "epochs": 160000,
        'batchsize': config['s2_batch_size'],
        }
    )

    """
    Model related
    """
    savepath = config['savepath']

    if local_rank == 0:
        # Logging
        if config['exp_id'].lower() != 'null':
            print('I will take the role of logging!')

            long_id = '%s' % (config['exp_id'])

        else:
            long_id = None
        logger = TensorboardLogger(config['exp_id'], long_id, git_info, False, savepath=savepath)
        logger.log_string('hyperpara', str(config))

        # Construct the rank 0 model
        model = ColorMNetTrainer(config, logger=logger, 
                        save_path=path.join(savepath, 'saves', long_id, long_id) if long_id is not None else None, 
                        local_rank=local_rank, world_size=world_size, wandb=wandb).train()
    else:
        # Construct model for other ranks
        model = ColorMNetTrainer(config, local_rank=local_rank, world_size=world_size).train()

    # Load pertrained model if needed
    if raw_config['load_checkpoint'] is not None:
        total_iter = model.load_checkpoint(raw_config['load_checkpoint'])
        raw_config['load_checkpoint'] = None
        print('Previously trained model loaded!')
    else:
        total_iter = 0

    if network_in_memory is not None:
        print('I am loading network from the previous stage')
        model.load_network_in_memory(network_in_memory)
        network_in_memory = None
    elif raw_config['load_network'] is not None:
        print('I am loading network from a disk, as listed in configuration')
        model.load_network(raw_config['load_network'])
        raw_config['load_network'] = None

    """
    Dataloader related
    """
    # To re-seed the randomness everytime we start a worker
    def worker_init_fn(worker_id): 
        worker_seed = torch.initial_seed()%(2**31) + worker_id + local_rank*100
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def construct_loader(dataset):
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
        train_loader = DataLoader(dataset, config['batch_size'], sampler=train_sampler, num_workers=config['num_workers'],
                                worker_init_fn=worker_init_fn, drop_last=True)
        return train_sampler, train_loader

    # valid for any dataset 
    def renew_DAVIS_Videvo_batch_loader(max_skip, finetune=False):
        # //5 because we only have annotation for every five frames
        davis_dataset = DAVISVidevoDataset(data_batch_root, 
                            data_ref_batch_root, max_skip, is_bl=False, subset=None, num_frames=config['num_frames'], finetune=finetune)
        train_dataset = ConcatDataset([davis_dataset])

        print(f'DAVIS + Videvo dataset size: {len(davis_dataset)}')
        print(f'Concat dataset size: {len(train_dataset)}')
        print(f'Renewed with {max_skip=}')

        return construct_loader(train_dataset)

    """
    Dataset related
    """

    """
    These define the training schedule of the distance between frames
    We will switch to max_skip_values[i] once we pass the percentage specified by increase_skip_fraction[i]
    Not effective for stage 0 training
    The initial value is not listed here but in renew_vos_loader(X)
    """
    max_skip_values = [10, 15, 5, 5]

    # stage 2
    increase_skip_fraction = [0.1, 0.3, 0.9, 100]
    # training set
    data_batch_root = raw_config['davis_root']
    data_ref_batch_root = data_batch_root

    val_data_batch_root = raw_config['validation_root']
    val_data_ref_batch_root = val_data_batch_root

    train_sampler, train_loader = renew_DAVIS_Videvo_batch_loader(5)
    renew_loader = renew_DAVIS_Videvo_batch_loader

    # validation 
    val_dataset = DAVISTestDataset_221128_TransColorization_batch(val_data_batch_root, imset=val_data_ref_batch_root)
    

    """
    Determine max epoch
    """
    total_epoch = math.ceil(config['iterations']/len(train_loader))
    current_epoch = total_iter // len(train_loader)
    print(f'We approximately use {total_epoch} epochs.')
    if stage != '0':
        change_skip_iter = [round(config['iterations']*f) for f in increase_skip_fraction]
        # Skip will only change after an epoch, not in the middle
        print(f'The skip value will change approximately at the following iterations: {change_skip_iter[:-1]}')

    """
    Starts training
    """
    finetuning = False
    # Need this to select random bases in different workers
    np.random.seed(np.random.randint(2**30-1) + local_rank*100)
    try:
        while total_iter < config['iterations'] + config['finetune']:
            
            # Crucial for randomness! 
            train_sampler.set_epoch(current_epoch)
            current_epoch += 1
            print(f'Current epoch: {current_epoch}')

            # Train loop
            model.train()
            for data in train_loader:
                # Update skip if needed
                if stage!='0' and total_iter >= change_skip_iter[0]:
                    while total_iter >= change_skip_iter[0]:
                        cur_skip = max_skip_values[0]
                        max_skip_values = max_skip_values[1:]
                        change_skip_iter = change_skip_iter[1:]
                    print(f'Changing skip to {cur_skip=}')
                    train_sampler, train_loader = renew_loader(cur_skip)
                    break

                # fine-tune means fewer augmentations to train the sensory memory
                if config['finetune'] > 0 and not finetuning and total_iter >= config['iterations']:
                    train_sampler, train_loader = renew_loader(cur_skip, finetune=True)
                    finetuning = True
                    model.save_network_interval = 1000
                    break

                model.do_pass(data, total_iter, val_dataset=val_dataset)
                total_iter += 1

                if total_iter >= config['iterations'] + config['finetune']:
                    break
    finally:
        if not config['debug'] and model.logger is not None and total_iter>5000:
            model.save_network(total_iter)
            model.save_checkpoint(total_iter)

    network_in_memory = model.XMem.module.state_dict()

    wandb.finish()

distributed.destroy_process_group()
