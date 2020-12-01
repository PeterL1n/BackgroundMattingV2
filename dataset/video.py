import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, path: str, transforms: any = None):
        self.cap = cv2.VideoCapture(path)
        self.transforms = transforms
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __len__(self):
        return self.frame_count
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = self.cap.read()
        if not ret:
            raise IndexError(f'Idx: {idx} out of length: {len(self)}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transforms:
            img = self.transforms(img)
        return img
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cap.release()
