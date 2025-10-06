module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh



export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

 python main_pretrain.py --model mae_vit_base_patch16 --data_path . --batch_size 64 --epochs 200 --num_workers 2 --input_size 224 --norm_pix_loss --output_dir ./output_mae_galaxy

  python main_linprobe.py --model mae_vit_base_patch16 --data_path . --batch_size 64 --epochs 200 --nb_classes 10 --num_workers 2 --output_dir ./lin_probe_test

  python main_linprobe.py --model vit_base_patch16 --data_path . --batch_size 8 --epochs 200 --nb_classes 10 --num_workers 2 --output_dir ./lin_probe_test --finetune ./output_mae_galaxy/checkpoint-0.pth
