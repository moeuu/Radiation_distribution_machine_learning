import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            transform=None,
        ) -> None:
            super().__init__()
            self.data_dir = data_dir
            self.transform = transform 
            self.image_files = os.listdir(data_dir)

    def __len__(self):
          return len(self.image_files)
    
    def __getitem__(self, idx):
          
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)

        return image