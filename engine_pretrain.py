# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torchvision.utils as vutils
import os

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)
        
        if data_iter_step % 500 == 0:  # every 500 iterations
            save_dir = os.path.join(log_writer.log_dir, "reconstructions")
            os.makedirs(save_dir, exist_ok=True)
            # Get reconstructed image
            reconstructed = model.unpatchify(pred.detach().cpu())

            # Get masked image visualization
            # 1 = masked, 0 = visible
            mask = mask.detach().cpu()
            img = samples.detach().cpu()

            # Convert patches to images for masking visualization
            N, C, H, W = img.shape
            p = model.patch_embed.patch_size[0]
            h = w = H // p

            mask = mask.unsqueeze(-1).repeat(1, 1, p**2 * 3)  # (N, L, p^2 * 3)
            mask = model.unpatchify(mask)  # (N, 3, H, W)
            masked_img = img * (1 - mask)  # Zero out masked patches

            
                        
            # pick one sample
            i = 0
            mask_vis = mask[i].clone()  # shape: (3, H, W)
            mask_vis = mask_vis.mean(dim=0, keepdim=True)  # collapse RGB to 1-channel grayscale
            mask_vis = mask_vis.repeat(3, 1, 1)  # expand back to 3 channels for saving
            original = img[i]
            masked = masked_img[i]
            recon = reconstructed[i]

            # concatenate horizontally
            grid = torch.cat([original, masked, mask_vis, recon], dim=2)  # side-by-side
            vutils.save_image(grid, os.path.join(save_dir, f"epoch{epoch:03d}_iter{data_iter_step:05d}.png"))
        

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}