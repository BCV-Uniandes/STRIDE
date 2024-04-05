# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import os

import torch
import torch.utils.data

import datasets.transforms as T
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import io
from PIL import Image
import numpy as np
import random


__all__ = ['build']

class AzureDetection(torch.utils.data.Dataset):
    def __init__(self, args, transforms):
        super(AzureDetection, self).__init__()
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()
        self.blob_service_client = BlobServiceClient(account_url=args.connection_string)
        self.container_name = args.container_name
        self.artifict_removal = T.ArtifactRemoval()
        self.load_images(args)
        
    def load_images(self, args):
        self.images_list = open(args.azure_list_path,'r').readlines()
        if os.path.isdir(os.path.join(args.output_dir,'Inferences')):
            infered_images = set(im.replace('.pth','.jpg') for im in os.listdir(os.path.join(args.output_dir,'Inferences')))
            self.images_list = [im for im in self.images_list if im not in infered_images]
    
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        # breakpoint()
        img = self.images_list[idx].strip()
        try:
            img = self.blob_service_client.get_blob_client(container=self.container_name, blob=img)
            img = img.download_blob().readall()
            img = Image.open(io.BytesIO(img))
            img = self.artifict_removal(img)
            image_id = idx
        except ResourceNotFoundError:
            image_id = -1
            img = np.random.rand(4000, 13312, 3) * 255
            img = img.astype(np.uint8)
            img = Image.fromarray(img, 'RGB')
            
        target = {'image_id': image_id}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


class ConvertCocoPolysToMask(object):
    def __init__(self):
        pass

    def __call__(self, image, target):
        w, h = image.size
        # breakpoint()
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        # guard against no boxes via resizing
        boxes = torch.as_tensor([], dtype=torch.float32).reshape(-1, 4)
        classes = torch.tensor([], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([])
        iscrowd = torch.tensor([])
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return T.Compose([
        T.RandomResize([args.img_size]),
        normalize,
    ])


def build(args):

    dataset = AzureDetection(args=args,transforms=make_coco_transforms(args=args), 
        )

    return dataset
