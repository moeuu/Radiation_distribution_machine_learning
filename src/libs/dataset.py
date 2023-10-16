import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(
            self, 
            data_dir,
            cor_dir,
            transform=None,
        ) -> None:
            super().__init__()
            self.data_dir = data_dir
            self.cor_dir = cor_dir
            self.transform = transform 
            self.image_files = os.listdir(data_dir)

    def __len__(self):
          return len(self.image_files)
    
    def __getitem__(self, idx):
          
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name)
        target_img_name = os.path.join(self.cor_dir, "cor" + str(idx)+ ".jpg")
        target_image_raw = Image.open(target_img_name)
        
        if self.transform:
            input_image = self.transform(image)
            target_image = self.transform(target_image_raw)

        return input_image, target_image