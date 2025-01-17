import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile
import pytorch_lightning as pl
import numpy as np
import torch
from comer.datamodule.dataset import CROHMEDataset
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader
from comer.datamodule.vocab import vocab
# from comer.curriculum.Cal_sample_loss import CalculateLoss
from collections import OrderedDict

Data = List[Tuple[str, Image.Image, List[str]]]

MAX_SIZE = 32e4  # change here accroading to your GPU memory

# load data
def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE,
    maxlen: int = 200,
    maxImagesize: int = MAX_SIZE,
    ):
    """
    return data as follow: 
    [
    ([fname1,fname2,fname3,...], [feature1,feature2,feature3,...], [label1,label2,label3,...]), 
    ([fname9,fname10,fname11,...], [feature9,feature10,feature11,...], [label9,label10,label11,...]), 
    ...]
    """
    fname_batch = []
    feature_batch = []
    label_batch = []

    fname_total = []
    feature_total = []
    label_total = []
    
    biggest_image_size = 0


    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])

    i = 0
    for fname, fea, lab in data:
        size = fea.size[0] * fea.size[1]
        fea = np.array(fea)
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[0]} x {fea.shape[1]} =  bigger than {maxImagesize}, ignore"
            )
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                label_batch = []
            fname_batch.append(fname)
            feature_batch.append(fea)
            label_batch.append(lab)
            i += 1
                
    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))


def extract_data(archive: ZipFile, dir_name: str) -> Data: # return data as follow: [(fname1, fea1, lab1), (fname2, fea2, lab2), ...]
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    with archive.open(f"data/{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    data = []
    for line in captions:
        tmp = line.decode().strip().split()
        img_name = tmp[0]
        formula = tmp[1:]
        with archive.open(f"data/{dir_name}/img/{img_name}.bmp", "r") as f:
            # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
            img = Image.open(f).copy()
        data.append((img_name, img, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

    # return fnames, x, x_mask, seqs_y
    return Batch(fnames, x, x_mask, seqs_y)


def build_dataset(archive, folder: str, batch_size: int):
    if folder != 'all':
        data = extract_data(archive, folder)
    else:
        data = []
        for folder in ['2014', '2016', '2019']:
            data += extract_data(archive, folder)
    return data_iterator(data, batch_size)

def build_curriculum_dataset(archive, folder: str, batch_size: int):
    if folder == 'train':
        simple_data = []
        medium_data = []
        hard_data = []
        data = [simple_data, medium_data, hard_data]
        captions = ['simple.txt','medium.txt','complex.txt']
        for i in range(len(captions)):
            data[i] = extract_data_train(archive, caption = captions[i])
            data[i] = data_iterator(data[i], batch_size)
    return data

def extract_data_train(archive: ZipFile, caption) -> Data: # return data as follow: [(fname1, fea1, lab1), (fname2, fea2, lab2), ...]
    with archive.open(f"classified_data/{caption}", "r") as f:
        captions = f.readlines()
    data = []
    for line in captions:
        tmp = line.decode().strip().split()
        img_name = tmp[0]
        formula = tmp[1:]
        with archive.open(f"data/train/img/{img_name}.bmp", "r") as f:
            # move image to memory immediately, avoid lazy loading, which will lead to None pointer error in loading
            img = Image.open(f).copy()
        data.append((img_name, img, formula))

    print(f"Extract data from: {caption}, with data size: {len(data)}")

    return data



class CL_CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.zipfile_path = self.config.data.zipfile_path
        self.test_year = self.config.data.test_year
        self.train_batch_size = self.config.data.train_batch_size
        self.eval_batch_size = self.config.data.eval_batch_size
        self.num_workers = self.config.data.num_workers
        self.scale_aug = self.config.data.scale_aug
        self.original_train_dataset = None

        print(f"Load data from: {self.zipfile_path}")

    def setup(
        self, 
        stage: Optional[str] = None,
        model = None) -> None:
        self.model = model

        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                # Train_dataset
                self.original_train_dataset = build_curriculum_dataset(archive, 'train',  self.train_batch_size)
                self.train_dataset = None
                # Val_dataset
                self.val_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size),
                    False,
                    self.scale_aug,
                )
            if stage == "test" or stage is None:
                # Test_dataset
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size),
                    False,
                    self.scale_aug,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
