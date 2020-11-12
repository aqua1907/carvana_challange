import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import torch


def get_train_val(path, split=0.2):
    """

    :param path: path to train_masks.csv
    :param split: validation split coef
    :return: return 2 DataFrames: train and validation
    """
    df = pd.read_csv(path)
    val_examples = int(len(df) * split)
    train_examples = len(df) - val_examples
    df = shuffle(df).reset_index(drop=True)

    train_df = df.iloc[:train_examples].copy()
    val_df = df.iloc[:val_examples].copy()
    del df

    return train_df, val_df


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def rle_decode(rle_str, shape):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(shape), dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1

    return mask.reshape(shape)


def predict(model, batch, device):
    preds = model(batch.to(device))
    preds = torch.sigmoid(preds).permute(0, 2, 3, 1)

    return preds.data.cpu().numpy()


