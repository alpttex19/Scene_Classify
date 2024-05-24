import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor


class Scene_Dataset(Dataset):
    def __init__(self, data_dir = "dataset",  mode="train"):
        imgs_root = f"{data_dir}/imgs"
        labels_root = f"{data_dir}/{mode}_data.csv"
        # 读取csv文件
        df = pd.read_csv(labels_root)
        imgs_names = df['image_name'].tolist()
        labels = df['label'].tolist()
        self.imgs = [pil_to_tensor(Image.open(f"{imgs_root}/{img_name}")) / 255 for img_name in imgs_names]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx].float(), self.labels[idx].long()


if __name__ == "__main__":
    data = Scene_Dataset()
    data_loader = DataLoader(data, batch_size=2, shuffle=True)
    for inputs, labels in data_loader:
        print(inputs.shape, labels)
        break