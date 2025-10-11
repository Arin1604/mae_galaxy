module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate mae38_env


export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

 python main_pretrain.py --model mae_vit_base_patch16 --data_path . --batch_size 64 --epochs 200 --num_workers 2 --input_size 224 --norm_pix_loss --output_dir ./output_dir/pre_train/output_mae_galaxy_reconstruction

 python main_pretrain.py --model mae_vit_base_patch16 --data_path . --batch_size 64 --epochs 200 --num_workers 2 --resume ./output_dir/pre_train/output_mae_galaxy_no_norm_pix_mask0_67/checkpoint-180.pth --input_size 224 --output_dir ./output_dir/pre_train/output_mae_galaxy_reconstruction

 ssl/mae_galaxy/

 python main_pretrain.py --model mae_vit_base_patch16 --data_path . --batch_size 64 --epochs 200 --num_workers 2 --input_size 224 --output_dir ./output_mae_galaxy_no_norm_pix

 python main_pretrain.py --model mae_vit_base_patch16 --data_path . --batch_size 64 --epochs 200 --num_workers 2 --input_size 224 --mask_ratio 0.67 --output_dir ./output_dir/pre_train/output_mae_galaxy_no_norm_pix_mask0_67

 python main_pretrain.py --model mae_vit_base_patch16 --data_path . --batch_size 64 --epochs 200 --num_workers 2 --resume ./output_dir/pre_train/output_mae_galaxy_no_norm_pix_mask0_67/checkpoint-180.pth --input_size 224 --mask_ratio 0.1 --output_dir ./output_dir/pre_train/output_mae_galaxy_reconstruction_vis_2

 python main_pretrain.py --model mae_vit_base_patch16 --data_path . --batch_size 64 --epochs 200 --num_workers 2 --input_size 224 --mask_ratio 0.65 --output_dir ./output_dir/pre_train/output_mae_galaxy_center_sampled_re_normalized_no_morm_pix_loss_mask_065

 python main_pretrain.py --model mae_vit_base_patch16 --data_path . --batch_size 64 --epochs 200 --num_workers 2 --input_size 224 --mask_ratio 0.85 --resume ./output_dir/pre_train/output_mae_galaxy_no_norm_pix_mask0_85/checkpoint-100.pth  --output_dir ./output_dir/pre_train/output_mae_galaxy_no_norm_pix_mask0_85

## Lin Probe
  python main_linprobe.py --model mae_vit_base_patch16 --data_path . --batch_size 64 --epochs 200 --nb_classes 10 --num_workers 2 --output_dir ./lin_probe_test

  python main_linprobe.py --model vit_base_patch16 --data_path . --batch_size 8 --epochs 200 --nb_classes 10 --num_workers 2 --output_dir ./lin_probe_no_norm_pix_pre_train --finetune ./output_dir/output_mae_galaxy_no_norm_pix/checkpoint-199.pth


  python main_linprobe.py --model vit_base_patch16 --data_path . --batch_size 16 --epochs 200 --nb_classes 10 --num_workers 2 --output_dir ./output_dir/lin_probe/lin_probe_no_norm_pix_pre_train_mask0_65_center_samp --finetune ./output_dir/pre_train/output_mae_galaxy_center_sampled_re_normalized_no_morm_pix_loss_mask_065/checkpoint-100.pth


  python main_linprobe.py --model vit_base_patch16 --data_path . --batch_size 8 --epochs 200 --nb_classes 10 --num_workers 2 --output_dir ./lin_probe_with_norm_pix_pre_train_batch64 --finetune ./output_mae_galaxy/checkpoint-199.pth

  python main_linprobe.py --model vit_base_patch16 --data_path . --batch_size 8 --epochs 200 --nb_classes 10 --num_workers 2 --output_dir ./lin_probe_with_norm_pix_pre_train_batch64 --finetune ./output_mae_galaxy/checkpoint-199.pth

    python main_linprobe.py --model vit_base_patch16 --data_path . --batch_size 32 --epochs 200 --nb_classes 10 --num_workers 2 --output_dir ./output_dir/lin_probe/augmented_lin_probe_no_norm_pix_pre_train_mask0_67 --finetune ./output_dir/pre_train/output_mae_galaxy_no_norm_pix_mask0_67/checkpoint-180.pth


python main_linprobe.py --model vit_base_patch16 --data_path . --batch_size 32 --epochs 200 --nb_classes 10 --num_workers 2 --output_dir ./output_dir/lin_probe/adam_augmented_lin_probe_no_norm_pix_pre_train_mask0_67 --finetune ./output_dir/pre_train/output_mae_galaxy_no_norm_pix_mask0_67/checkpoint-180.pth
  

