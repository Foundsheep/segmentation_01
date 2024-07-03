import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import v2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

from utils.preprocess import erase_coloured_text_and_lines


def load_data(root, is_train, shuffle, batch_size, num_workers):
    ds = SPRDataset(root, is_train)
    dl = DataLoader(ds,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    persistent_workers=True)
    return dl


class SPRDataset(Dataset):
    """
    1. root 폴더 아래에 'annotated', 'raw' 폴더가 있는 것을 가정한다.
    2. 'annotated' 폴더 안에 'labelmap.txt' 파일로서 segmentation mask 이미지의 값과 라벨을
       설명하는 텍스트 파일이 있는 것을 가정한다.
    3. 'raw' 폴더 내 파일명과 'annotated' 폴더 내 파일명이 동일하다고 가정한다.(확장자 제외)
       -> cvat을 이용한 마스크 이미지의 확장자는 항상 png여서 'raw' 폴더 내 이미지
          확장자가 jpg인 경우 확장자만 바꾼 파일명을 사용하여 탐색한다.
    """
    def __init__(self, root, is_train=True):
        super().__init__()
        self.root = root
        self.is_train = is_train
        self.transforms = self._get_transforms()
        self.img_list, self.label_list, self.label_txt = self._read_paths()
        self.num_classes = len(self.label_txt)
        self.label_to_name_map, self.name_to_label_map = self._make_mapping_dict()
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label_path = self.label_list[idx]

        # read image
        img = Image.open(img_path)
        img = np.array(img)
        label_rgb = Image.open(label_path)
        label_rgb = np.array(label_rgb)

        label_mask = np.zeros((label_rgb.shape[0], label_rgb.shape[1], self.num_classes))

        # convert label to multi dimensional [0, 1] valued image
        for txt_idx, txt in enumerate(self.label_txt):
            divider_1 = txt.find(":")
            divider_2 = txt.find("::")

            label_name = txt[:divider_1]
            label_value = txt[divider_1+1:divider_2]
            
            rgb_values = list(map(int, label_value.split(",")))

            x, y = np.where(
                        (
                            (label_rgb[:, :, 0] == rgb_values[0]) & 
                            (label_rgb[:, :, 1] == rgb_values[1]) & 
                            (label_rgb[:, :, 2] == rgb_values[2])
                        )
                    )
            label_mask[x, y, txt_idx] = 1

        if self.transforms:
            augmented = self.transforms(image=img, mask=label_mask)
            img = augmented["image"]
            label_mask = augmented["mask"]
        return img, label_mask
    
    def _read_paths(self):
        folder_raw = Path(self.root) / Path("raw")
        folder_preprocessed = Path(self.root) / Path("preprocessed")
        folder_annotated = Path(self.root) / Path("annotated")

        # if annotated folder doesn't exist, it doesn't process
        if not folder_annotated.exists():
            print("annotated folder doesn't exist, so returning None")
            return None, None, None

        # if pre-processed data already exist
        if folder_preprocessed.exists() and len(list(folder_preprocessed.glob("*"))) > 0:
            print(f"preprocessed folder exists")
        
        else:
            # TODO: * is used here. Might be better with .jpg and .png selectively
            glob_raw = folder_raw.glob("*")

            if not folder_preprocessed.exists():
                folder_preprocessed.mkdir()
                print(f"preprocesed folder: [{str(folder_preprocessed)}] is made")
            
            for raw_file_name in glob_raw:
                img_preprocessed = erase_coloured_text_and_lines(str(raw_file_name))
                img_name = raw_file_name.parts[-1]
                path_preprocessed = folder_preprocessed / Path(img_name)
                cv2.imwrite(str(path_preprocessed), img_preprocessed)
        
        img_list = list(folder_preprocessed.glob("*.png")) + list(folder_preprocessed.glob("*.jpg"))
        # label_list = list(folder_annotated.glob("*.png")) + list(folder_annotated.glob("*.jpg"))
        label_list = [folder_annotated / Path(p.parts[-1] if p.parts[-1].endswith(".png") else p.parts[-1][:-4] + ".png") for p in img_list]

        # to convert label image to multi dimensional [0,1] valued image
        label_map_path = folder_annotated / Path("labelmap.txt")
        with open(str(label_map_path), "r") as f:

            # first and second lines are asuumed to have title and background info
            label_txt = f.readlines()[2:]
        
        return img_list, label_list, label_txt

    def _make_mapping_dict(self):
        label_to_name = {}
        name_to_label = {}

        for txt_idx, txt in enumerate(self.label_txt):
            divider_1 = txt.find(":")
            label_name = txt[:divider_1]

            label_to_name[txt_idx] = label_name
            name_to_label[label_name] = txt_idx

        return label_to_name, name_to_label
    
    def _get_transforms(self):
        if self.is_train:
            transforms = A.Compose([
                A.Resize(height=32*12, width=32*18),
                A.HorizontalFlip(),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(transpose_mask=True),
            ])
        else:
            transforms = A.Compose([
                A.Resize(height=32*12, width=32*18),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(transpose_mask=True),
            ])
        return transforms 


if __name__ == "__main__":
    path = "./datasets/spr_sample_01"
    print(Path(path).exists())

    dl = load_data(path, is_train=True, shuffle=True, batch_size=3)

    for d in dl:
        break
    
    print("=== train")
    print("image size: ", d[0].size())
    print("mask size: ", d[1].size())

    dl = load_data(path, is_train=False, shuffle=True, batch_size=3)

    for d in dl:
        break

    print("=== valid")
    print("image size: ", d[0].size())
    print("mask size: ", d[1].size())
