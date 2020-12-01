from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, dataset, samples):
        samples = min(samples, len(dataset))
        self.dataset = dataset
        self.indices = [i * int(len(dataset) / samples) for i in range(samples)]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
