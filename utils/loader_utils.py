
import os
import cv2
import random
import numpy as np
from PIL import Image
 
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms, utils
import random
def get_stamp_list(dataset, timestamp):
    base_dataset = getattr(dataset, "dataset", None)
    if base_dataset is None:
        raise AttributeError("Expected dataset to expose the underlying data via `.dataset`.")

    if hasattr(base_dataset, "poses"):
        pose_count = len(base_dataset.poses)
    else:
        # Fallback for dataset variants (e.g., Technicolor) providing a plain list of CameraInfo.
        if len(base_dataset) == 0:
            raise ValueError("Cannot build timestamp list from an empty dataset.")
        camera_ids = []
        for item in base_dataset:
            image_name = getattr(item, "image_name", None)
            if not isinstance(image_name, str):
                continue
            camera_ids.append(image_name.split("/")[0])
        if not camera_ids:
            raise AttributeError("Unable to infer unique camera ids from dataset.")
        pose_count = len(set(camera_ids))

    if pose_count == 0:
        raise ValueError("Pose count inferred as zero; cannot compute timestamp slices.")

    frame_length = int(len(dataset) / pose_count)
    if frame_length == 0:
        raise ValueError("Frame length inferred as zero; check dataset ordering.")

    if len(dataset) % pose_count != 0:
        print(f"[WARN] Dataset length {len(dataset)} not divisible by camera count {pose_count}.")

    if timestamp >= frame_length:
        raise IndexError("input timestamp bigger than total timestamp.")

    indices = [i * frame_length + timestamp for i in range(pose_count)]
    return [dataset[idx] for idx in indices]


class FineSampler(Sampler):
    def __init__(self, dataset):
        self.len_dataset = len(dataset) 
        self.len_pose = len(dataset.dataset.poses)
        self.frame_length = int(self.len_dataset/ self.len_pose)

        sample_list = []
        for i in range(self.frame_length):
            for j in range(4):
                idx = torch.randperm(self.len_pose) *self.frame_length + i
                now_list = []
                cnt = 0
                for item in idx.tolist():
                    now_list.append(item)
                    cnt+=1
                    if cnt % 2 == 0 and len(sample_list)>2:    
                        select_element = [x for x in random.sample(sample_list,2)]
                        now_list += select_element
            
            sample_list += now_list
            
        self.sample_list = sample_list
    def __iter__(self):

        return iter(self.sample_list)
    
    def __len__(self):
        return len(self.sample_list)
