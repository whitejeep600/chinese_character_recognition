from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import get_target_device


class RecognitionDataset(Dataset):
    def __init__(self, data: List[Dict], label_mapping: Dict[str, int],):
        self.data = data
        self.label_mapping = label_mapping
        self.target_device = get_target_device()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        return self.data[index]

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples):
        image_tensor = torch.stack([image['image'] for image in samples]).to(self.target_device)
        label_tensor =\
            torch.LongTensor([self.label_mapping[image['label']] for image in samples]).to(self.target_device)
        return({'image': image_tensor,
                'label': label_tensor})
