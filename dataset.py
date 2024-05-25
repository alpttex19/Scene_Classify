import os
import torch
import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor

# 自己定义的数据集类
class Scene_Dataset(Dataset):
    def __init__(self, data_dir = "dataset",  mode="train", transform=None):
        self.imgs_root = f"{data_dir}/imgs"
        # 读取csv文件
        labels_root = f"{data_dir}/{mode}_data.csv"
        df = pd.read_csv(labels_root)
        self.imgs_names = df['image_name'].tolist()
        labels = df['label'].tolist()
        self.labels = torch.tensor(labels)
        # 数据增强或者转换
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = cv2.imread(f"{self.imgs_root}/{self.imgs_names[idx]}")
        if self.transform :
            image = self.transform(image)
        return image.float(), self.labels[idx].long()

# 用来测试数据集是否能够正常运行
if __name__ == "__main__":
    data = Scene_Dataset()
    data_loader = DataLoader(data, batch_size=2, shuffle=True)
    for inputs, labels in data_loader:
        print(inputs.shape, labels)
        break