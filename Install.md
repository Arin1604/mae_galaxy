## Python and Dependencies

For the MAE codebase, I ended up using Python 3.8.

(Initially, I tried upgrading the code base to the laltest version of each dependency but this proved difficult as certain dependencies relied on very specifc versions of other dependencies. Particularly, timm and numpy. To get around this, I keep all dependnecies at the version MAE expects.)

### Dependencies:
The required dependencies and their versions are listed in requirements.txt. I suggest creating a Python venv using these requirements to run the code.

A Conda environment can also be made using the same dependecies if running on Oscar. I ran this repo on Oscar using the conda environment and on my local machine using the Python venv. Both should work. As a reminder, you will need to install Python 3.8 for this code to run.

```bash
# Create
python3.8 -m venv mae_venv

# Activate 
source mae_venv/bin/activate
# Windows
mae_venv\Scripts\activate

# Check Python version
python --version

pip install -r requirements.txt
```

For Conda:

```bash
conda create -n mae_venv python=3.8

# Activate 
conda activate mae_venv

pip install -r requirements.txt
```

## Dataset Loading

The script should automatically handle loading the Galaxy dataset from hugging face. Running the script for the first time may take some time as it will download the dataseet. Following the download, I wrap the Dataset in a PyTorch Dataset and it is then fed into the data loader.

## Running Instructions

The pre-train command I recommend using:

You can set the --center_masking flag to sample the masks with a center bias. You can remove it to go back to regular random sampling

```bash
python main_pretrain.py --model mae_vit_base_patch16 --center_masking --data_path . --batch_size 64 --epochs 200 --num_workers 2 --input_size 224 --mask_ratio 0.65 --output_dir ./output_dir/pre_train/output_mae_galaxy_065
```

Pass the path to your checkpoint after the --finetune argument.

```bash
python main_linprobe.py --model vit_base_patch16 --data_path . --batch_size 32 --epochs 200 --nb_classes 10 --num_workers 2 --output_dir ./output_dir/lin_probe/adam_augmented_lin_probe_no_norm_pix_pre_train_mask0_67 --finetune ./output_dir/pre_train/output_mae_galaxy_no_norm_pix_mask0_67/checkpoint-180.pth
```