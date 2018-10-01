from itertools import chain

import glob
import torch
import os
from PIL import Image
from os import path
from torch.utils.data import Dataset
import pdb
import numpy as np
import cv2


class SegmentationDataset(Dataset):
    _EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

    def __init__(self, in_dir, transform):
        super(SegmentationDataset, self).__init__()

        self.in_dir = in_dir
        self.transform = transform

        # Find all images
        self.images = []
        for img_path in chain(*(glob.iglob(path.join(self.in_dir, ext)) for ext in SegmentationDataset._EXTENSIONS)):
            _, name_with_ext = path.split(img_path)
            idx, _ = path.splitext(name_with_ext)
            self.images.append({
                "idx": idx,
                "path": img_path
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Load image
        with Image.open(self.images[item]["path"]) as img_raw:
            size = img_raw.size
            img = self.transform(img_raw.convert(mode="RGB"))

        return {"img": img, "meta": {"idx": self.images[item]["idx"], "size": size}}


def segmentation_collate(items):
    imgs = torch.stack([item["img"] for item in items])
    masks = torch.stack([torch.from_numpy(item["target"]) for item in items])
    metas = [item["meta"] for item in items]

    return {"img": imgs, "meta": metas, "target": masks}


class TrainingSegmentationDataset(SegmentationDataset):
    
    def __init__(self,in_dir, transform,masks_dir):
        super(TrainingSegmentationDataset,self).__init__(in_dir, transform)
        self.masks = []
        self.masks_dir = masks_dir

        for npy_path in os.listdir(path.join(self.masks_dir)):
            #_, name_with_ext = path.split(npy_path)
            #idx, _ = path.splitext(name_with_ext)
            #pdb.set_trace()

            self.masks.append(path.join(self.masks_dir,npy_path))
            '''
            self.masks.append({
                "idx": idx,
                "path": npy_path
            })
            '''
        self.images = sorted(self.images,key=lambda x: (x["idx"]))
        self.masks = sorted(self.masks)

    def __getitem__(self, item):
        # Load image
        
        with Image.open(self.images[item]["path"]) as img_raw:
            size = img_raw.size
            img = self.transform(img_raw.convert(mode="RGB"))
        

        m = np.load(self.masks[item])
        #pdb.set_trace()
        
        return {"img": img, "target": np.resize(m,(2048,2048)),
                "meta": {"idx": self.images[item]["idx"], "size": size}}


