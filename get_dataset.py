from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset

import torch
print(torch.version.cuda)   # prints CUDA version PyTorch was built with
print(torch.cuda.is_available())  # True if a GPU is detected




#i wanna now use this dataset to train the MAE model

#train, test = ds['train'].train_test_split(test_size=0.2, shuffle=True)

#use this train and test data to initialize your models
from torch.utils.data import Dataset

class GalaxyDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        # store the HuggingFace dataset
        self.dataset = hf_dataset
        # store the transform (e.g. Resize, Normalize)
        self.transform = transform

    def __len__(self):
        # return how many samples are in the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # grab one sample
        sample = self.dataset[idx]
        img = sample["image"]     # a PIL Image
        label = sample["label"] # an int

        # apply transforms if given
        if self.transform:
            img = self.transform(img)

        return img, label

class Galaxy_Dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def init_loaders(self):
        ds = load_dataset(self.dataset_path)
        print(ds["train"])
        # ds["train"][0]["image"].show()

        ds = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])


        #we need to modify the dataset, can I apply mods on the entire ds object?



dataset = Galaxy_Dataset("matthieulel/galaxy10_decals")
dataset.init_loaders()
