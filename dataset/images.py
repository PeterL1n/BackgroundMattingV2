import os
from torch.utils.data import Dataset
from PIL import Image

def _get_filenames(path):
    imgs = []
    valid_images = [".jpg", ".jpeg", ".png"]
    for file_path in os.listdir(path):
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in valid_images:
            continue
        imgs.append(os.path.join(path, file_path))
    return imgs

class ImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.filenames = sorted(_get_filenames(root))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)
        
        if self.transforms:
            img = self.transforms(img)
        
        return img
