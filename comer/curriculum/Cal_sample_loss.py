from torch.utils.data import DataLoader
import torch
from comer.utils.utils import to_bi_tgt_out, ce_loss
from comer.datamodule.dataset import CROHMEDataset
from comer.datamodule.vocab import vocab
from comer.datamodule.datamodule import collate_fn
from collections import OrderedDict
import numpy as np

MAX_SIZE = 32e4 
def data_iterator(
    data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE,
    maxlen: int = 200,
    maxImagesize: int = MAX_SIZE,
    check_ignore = False,
): """
# return data as follow: 
# [
# ([fname1,fname2,fname3,...], [feature1,feature2,feature3,...], [label1,label2,label3,...]), 
# ([fname9,fname10,fname11,...], [feature9,feature10,feature11,...], [label9,label10,label11,...]), 
# ...]
"""
    fname_batch = []
    feature_batch = []
    label_batch = []

    fname_total = []
    feature_total = []
    label_total = []
    
    biggest_image_size = 0

    ignore = []

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
            if check_ignore: 
                ignore.append((fname, fea, lab))
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                fname_total.append(fname_batch) # fname_total = [[fname1,fname2,...],[],[],...]
                feature_total.append(feature_batch) # feature_total = [[feature1,feature2,...],[],[],...]
                label_total.append(label_batch) # label_total = [[label1,label2,...],[],[],...]
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
    return list(zip(fname_total, feature_total, label_total)), ignore 

class CalculateLoss:
    def __init__(self,model, dataset) -> None:
        
        self.dataset_len = len(dataset)
        raw_data, ignore = data_iterator(dataset, batch_size=1,check_ignore = True)
        self.ignore = ignore
        self.dataset = CROHMEDataset(
            ds = raw_data,
            is_train = False,
            scale_aug = False
            )
        self.model = model
        self.sample_losses = {}
        self.sort_by_lenght = {}

    def Calculate_sample_loss(self):
        """Calculate loss for samples"""
        print("Calculating initial losses for all samples in dataset...")
        print("Data currently: ", self.dataset_len)
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers= 5,
            collate_fn= collate_fn
        )

        # Calculate loss for each sample
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                batch.imgs = batch.imgs.to(self.model.device)
                batch.mask = batch.mask.to(self.model.device)
                
                tgt, out = to_bi_tgt_out(batch.indices, self.model.device)
                out_hat = self.model(batch.imgs, batch.mask, tgt)
                loss = ce_loss(out_hat, out)

                img_name = batch.img_bases[0]
                
                self.sort_by_lenght[img_name] = len(img_name)
                self.sample_losses[img_name] = loss.item()

            self.sort_by_lenght = OrderedDict(sorted(self.sort_by_lenght.items(), key=lambda x: x[1]))
            
            self.sample_losses = OrderedDict(sorted(self.sample_losses.items(), key=lambda x: x[1]))
                
                if idx % 100 == 0:
                    print(f"Processed {idx}/{self.dataset_len} samples")
                
        print(f"Calculated losses for {len(self.sample_losses)} samples")

        