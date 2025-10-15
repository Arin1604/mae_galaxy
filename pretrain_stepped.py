#The goal of this script is to pretrain a MAE Vision Backbone
# Crucially, this will also frequently analyze the model at certain weights

#write a parser, borrowing some of the parser code from pre-existing code

#general python deps for file handling, etc.
import argparse
import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

#Hugging face dataset dependencies, to load hfs datasets specifically
import datasets
from datasets import load_dataset

#custom dataset wrapper
from get_dataset import HFSDataset
from transform_generate import generate_transforms

#torch dependencies for handling models and tensors
import torch
import torch.backends.cudnn as cudnn

import numpy as np


#torch vision dependencies
import torchvision.transforms as transforms


def arg_parser_init():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--center_masking', action='store_true',
                    help='enable center masking')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    #linear probing parameters
    parser.add_argument('--lin_probe', action='store_true', help='Do linear Probing During Pre-training')

    return parser

def main(args):
    '''
    Things that will happen in our pretraining loop in order
    1) Initialize Distributed parameters, implement later under these better
    2) Load the hfs dataset
    '''

    #print current dir,
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print("Outputs will be stored in {}".format(args.output_dir))

    #sets the GPU/CPU for our torch
    device = torch.device(args.device)

    # fix the seed for reproducibility
    #this makes sure that the initial set of random weights and biases are deterministic
    #seed = args.seed + misc.get_rank()
    #hard code for time being
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    #Learn more about this!
    cudnn.benchmark = True

    #generate transforms here
    ##########################################################################
    pre_train_transforms, lin_probe_train_transforms, lin_probe_val_transforms = generate_transforms()
    

    #Load the hfs dataset and apply our transforms
    hfs_dataset = load_dataset("matthieulel/galaxy10_decals")
    split = hfs_dataset['train'].train_test_split(0.2)

    #maybe make these much smaller?
    lin_train = split['train']
    lin_test = hfs_dataset['test']

    #3 datasets, 1 for pre-training our MAE, 1 for training linear probe, 1 for validating it
    pretrain_dataset = HFSDataset(hfs_dataset['train'], transform=pre_train_transforms)
    lin_probe_train_dataset = HFSDataset(lin_train, transform=lin_probe_train_transforms)
    lin_probe_validation_dataset = HFSDataset(lin_test, transform=lin_probe_val_transforms)


    #define a sampler:
    sampler_pre_train = torch.utils.data.RandomSampler(pretrain_dataset)
    sampler_lin_probe_train = torch.utils.data.RandomSampler(lin_probe_train_dataset)
    sampler_lin_probe_val = torch.utils.data.SequentialSampler(lin_probe_validation_dataset)

    if args.log_dir is None:
        args.log_dir = os.path.join(args.output_dir, "./logs")

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    #3 dataloaders for our three datasets
    data_loader_pretrain = torch.utils.data.DataLoader(
        pretrain_dataset, sampler=sampler_pre_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_lin_probe_train = torch.utils.data.DataLoader(
        lin_probe_train_dataset, sampler=sampler_lin_probe_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_lin_probe_val = torch.utils.data.DataLoader(
        lin_probe_validation_dataset, sampler=sampler_lin_probe_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )



if __name__ == '__main__':
    args = arg_parser_init()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


