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
import numpy
import sys
from typing import Iterable
from PIL import Image
import os
import torchvision.transforms as T



import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, tmp_out_dir="./tmp"):
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
        # print("This is samples")
        # print(samples.size())
        

        with torch.cuda.amp.autocast():
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)
        
        # print("This is pred")
        # print(pred.size())
        # print("This is mask")
        # print(mask.size())

        if epoch == 0 and data_iter_step % 100 == 0:
            # take first image in batch
            img = samples[0].detach().cpu()
            #this is rn a tensorflow tensor
            unnormalize = T.Normalize(
                mean=[-m/s for m, s in zip([0.5,0.5,0.5],[0.5,0.5,0.5])],
                std=[1/s for s in [0.5,0.5,0.5]]
            )
            # img = unnormalize(img)
            img = torch.clamp(img, 0, 1)

            # convert tensor to HWC numpy
            img = T.ToPILImage()(img)

            # save
            out_path = os.path.join(tmp_out_dir, f"epoch{epoch}_step{data_iter_step}.png")
            img.save(out_path)
            print(f"Saved sample image to {out_path}")

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