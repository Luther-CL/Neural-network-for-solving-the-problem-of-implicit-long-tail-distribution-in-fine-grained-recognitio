from PIL import Image
import torch
import os
import json
import pandas as pd
from torch.utils.data import Dataset

json_path = "./class/cub_132.json"
with open(json_path, 'r') as load_f:
    load_data_dict = json.load(load_f)

class MyDataSet(Dataset):
    """自定义数据集"""
    def __init__(self, images_path: list, images_class: list, transform=None, flag="train"):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])

        if img.mode != 'RGB':
            img = img.convert('RGB')

        label_fine = self.images_class[item]
        label_coarse = self.build_mapping(label_fine)
        assert label_coarse is not None, "The value of label_coarse is wrong! "

        if self.transform is not None:
            img = self.transform(img)

        return img, label_coarse, label_fine

    def build_mapping(self, image_class_index):
        return_value = None
        items = load_data_dict.items()
        for i, v in enumerate(items):
            cla_name, cla_indices = v[0], v[1]
            if image_class_index in cla_indices:
                return_value = i
                break
        # if return_value is None:
        #     print(f"!!!!!!!!!!!!!!!!!!!!!!!{image_class_index}!!!!!!!!!!!")
        return return_value

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
