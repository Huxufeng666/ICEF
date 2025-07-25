from PIL import Image
import os 
from torch.utils.data import Dataset

from torchvision import transforms

class get_data (Dataset):
    def __init__(self, image_dir:str,mask_dir:str,image_transform=None,mask_transform=None):
        
        self.image_dir= image_dir
        self.mask_dir = mask_dir        
        self.image_paths = sorted(os.listdir(image_dir))
        
        
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.image_transform = image_transform
            
        if mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x > 0.5).float())
            ])
        else:
            self.mask_transform = mask_transform


    def  __len__(self):
        return len(self.image_paths)
    
    
    def __getitem__(self,idx):
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.image_dir,image_name)
        mask_path = os.path.join(self.mask_dir,image_name)
        image = Image.open(image_path).convert("L")

        
        base_name , ext = os.path.splitext(image_name)
        mask_candidate1 = os.path.join(self.mask_dir,f"{base_name}_mask_1.png")
        mask_candidate2 = os.path.join(self.mask_dir,f"{base_name}_mask.png")
        if os.path.exists(mask_candidate1):
            mask_path = mask_candidate1
        elif os.path.exists(mask_candidate2):
            mask_path = mask_candidate2
        
        else:
            raise FileExistsError(f"not{base_name} mask")
        
        mask = Image.open(mask_path).convert("L")
        
      
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        return image,mask

