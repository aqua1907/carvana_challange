import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.scripts import rle_decode
from utils.transforms import ToTensor, Compose, Rescale, RandomCrop
import cv2
from torchvision.transforms import functional as T


class CarvanaDataset(Dataset):
    def __init__(self, df, root_dir, transform):
        self.dataframe = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.root_dir / self.dataframe.iloc[idx, 0]
        img = cv2.imread(str(img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        rle_str = self.dataframe.iloc[idx, 1]
        mask = rle_decode(rle_str, (h, w, 1))
        mask = mask.astype(np.uint8)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        sample = {"image": img, "mask": mask}

        return sample


class CarvanaEvalDataset(Dataset):
    def __init__(self, df, root_dir):
        self.dataframe = df
        self.root_dir = root_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.root_dir / self.dataframe.iloc[idx, 0]
        img = cv2.imread(str(img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        img = T.to_tensor(img)

        return {"image": img}


def get_transform(train):
    transforms = []
    if train:
        transforms.append(Rescale(256))
        transforms.append(RandomCrop(224))
    else:
        transforms.append(Rescale(224))
    transforms.append(ToTensor())

    return Compose(transforms)


def get_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    dataloder = DataLoader(dataset, batch_size=batch_size,
                           shuffle=shuffle, num_workers=num_workers)

    return dataloder
