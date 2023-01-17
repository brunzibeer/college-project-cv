import torch
from torch.utils.data import DataLoader, random_split, Dataset

BATCH_SIZE=8

class BigDataset(Dataset):
    """Dataset for handling big quantities of data.

    Extends the class :class:`Dataset`

    This class avoids to load all the data in memory at the same time, but loads only the necessary ones.
    The dataset folder must be contain file with only single tensors.
    File name must be the index of the tensor and must be formatted as "{index}_clip.pt"

    Args:
        labels (:class:`Tensor`): Tensor containing all labels. Shape = (num_elements)

        path (:class:`str`): Path of the folder that contains the dataset files.

        transform: Transformations on the image red from the dataset.
    """
    def __init__(self, labels, path, transform = None):
        'Initialization'
        self.labels = labels
        self.list_IDs = range(len(labels))
        self.path = path
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load(f'{self.path}/{ID}_clip.pt').permute(0, -1, 1, 2).float().div(255)     
        #print(f"DT: max: {X.max()}; min: {X.min()}")
        if self.transform:
            X = self.transform(X)
        y = self.labels[ID].long()
        return X, y

class UnNormalize(object):
    """Unormalize a torch tensor
        
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for frame in tensor:
            for t, m, s in zip(frame, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return tensor

def splitDataset(dataset, train_perc = 0.7, eval_perc=0.15):
    N = len(dataset)
    n_train = round(N*train_perc)
    n_eval = round(N*eval_perc)
    n_test = N - n_train - n_eval
    train, dev, test, = random_split(dataset, [n_train, n_eval, n_test])
    return train, dev, test

def createDataLoader(dataset, batch_size=BATCH_SIZE):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)