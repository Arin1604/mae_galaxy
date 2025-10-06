from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset

import torch
print(torch.version.cuda)   # prints CUDA version PyTorch was built with
print(torch.cuda.is_available())  # True if a GPU is detected


#i wanna now use this dataset to train the MAE model
#note that this is a hugging face dataset, we need to wrap it in a pytrch dataset
#for this purpose, we must implement the get item function

#this is just a wrapper around the hf data dict, we can apply transforms to things if needed

#train, test = ds['train'].train_test_split(test_size=0.2, shuffle=True)

#use this train and test data to initialize your models
from torch.utils.data import Dataset

class Galaxy_Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        #for a given index it should give the index and the label
        img = self.dataset[index]["image"]
        label = self.dataset[index]["label"]

        if self.transform:
            img = self.transform(img)

        return img, label

# hf_dict = load_dataset("matthieulel/galaxy10_decals")
# galaxy_data = Galaxy_Dataset(hf_dict["train"])

class Galaxy_Dataset_Loader:
    def __init__(self, dataset, seed=None, transforms=None, batch_size=64):
        pass
        #get dataset dict
        galaxy_dataset = load_dataset("matthieulel/galaxy10_decals")
        print(galaxy_dataset)
        self.transforms = transforms
        self.batch_size = batch_size
        split = galaxy_dataset["train"].train_test_split(test_size=0.2, seed=seed)
        self.train = split["train"]
        self.val   = split["test"]
        self.test = galaxy_dataset["test"]
        
        print(self.train)
        print(self.test)
        print(self.val)

    def get_train_loader(self):
        train_dataset = Galaxy_Dataset(self.train, transform=self.transforms)
        #wth is a sampler train?
        sampler_train = torch.utils.data.RandomSampler(train_dataset)




dataloader = Galaxy_Dataset_Loader()





























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

# class Galaxy_Dataset(Dataset):
#     def __init__(self, dataset_path):
#         self.dataset_path = dataset_path

#     def init_loaders(self):
#         ds = load_dataset(self.dataset_path)
#         print(ds["train"])
#         # ds["train"][0]["image"].show()

#         ds = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
#         ])


        #we need to modify the dataset, can I apply mods on the entire ds object?



# dataset = Galaxy_Dataset("matthieulel/galaxy10_decals")
# dataset.init_loaders()
