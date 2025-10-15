import torchvision.transforms as transforms
import torch

def generate_transforms(args):
    #perform transforms that center the galaxy. Also maybe play around with Image Formats
    pre_train_transform = transforms.Compose(
        [transforms.CenterCrop(184),
         transforms.Resize(224),
         transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.log1p(x)),
        transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])]
    )

    lin_probe_transform_train = transforms.Compose([
            transforms.CenterCrop(180),          # crop smaller region
            transforms.Resize(224), 
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    lin_probe_transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return pre_train_transform, lin_probe_transform_train, lin_probe_transform_val