import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class ImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.filenames = sorted([*glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True),
                                 *glob.glob(os.path.join(root, '**', '*.png'), recursive=True)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)
        
        if self.transforms:
            img = self.transforms(img)
        
        return img
