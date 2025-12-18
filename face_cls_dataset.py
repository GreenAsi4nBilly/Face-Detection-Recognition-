from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch
import glob
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

class FaceDataset(Dataset):
    def __init__(self, root , transform = None):
        self.transform = transform
        self.categories = ['Female','Male']
        self.images_path = []
        self.labels = []
        data_path = os.path.join(root)
        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for item in os.listdir(data_files):
                path = os.path.join(data_files,item)
                self.images_path.append(path)
                self.labels.append(i)
        # print(self.images_path)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label
    
if __name__ == '__main__':
    root = "dataset"
    dataset = FaceDataset(root)
    print(dataset.__len__())
    index = 32
    image,label = dataset.__getitem__(index)
    print(label)
    image.show()
        