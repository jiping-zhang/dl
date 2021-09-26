import random
from random import randint
import numpy as np
import cv2
from tqdm import tqdm
from abc import abstractmethod
from typing import *


def shuffle(array: list) -> None:
    length = len(array)
    for i in range(length):
        j = randint(0, length-1)
        temp = array[i]
        array[i] = array[j]
        array[j] = temp


class Dataset:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> tuple:
        raise NotImplementedError


class ListDataset(Dataset):
    def __init__(self, datas: List[tuple]) -> None:
        super().__init__()
        self.datas = datas

    def __len__(self) -> int:
        return len(self.datas)

    def __getitem__(self, index: int) -> tuple:
        return self.datas[index]


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, drop_last: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.dset_total_len = len(dataset)
        self.dset_round_len = (self.dset_total_len//batch_size)*batch_size if drop_last else self.dset_total_len
        if self.dset_round_len <= 0:
            raise ValueError("There are too few data !")
        self.single_data_len = len(list(dataset[0]))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.iter_order = None
        self.__start_new_round()

    def __start_new_round(self) -> None:
        iter_order = [i for i in range(len(self.dataset))]
        if self.shuffle:
            shuffle(iter_order)
        else:
            iter_order.reverse()
        if self.dset_total_len != self.dset_round_len:
            iter_order = iter_order[:self.dset_round_len]
        self.iter_order = iter_order

    def __get_batch(self) -> tuple:
        batch = []
        size_this = min(self.batch_size, len(self.iter_order))
        for i in range(self.single_data_len):
            batch.append([0]*size_this)
        for i in range(size_this):
            single_data = list(self.dataset[self.iter_order.pop()])
            for j in range(self.single_data_len):
                batch[j][i] = single_data[j]
        return tuple(batch)

    def __len__(self) -> int:
        return self.dset_total_len//self.batch_size

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        if len(self.iter_order) == 0:
            self.__start_new_round()
            raise StopIteration
        return self.__get_batch()


class MNIST(Dataset):
    def __init__(self, path: str, train: bool = True, flatten_x=False, size: int = 60000, strengthen: bool = False, *_, strengthenCount: int = 9) -> None:
        super().__init__()
        def generate_y_label(num: int) -> List[float]:
            ans = [0.0]*10
            ans[num] = 1.0
            return ans
        MNIST_IMG_SIZE = 28
        if flatten_x:
            shape = MNIST_IMG_SIZE*MNIST_IMG_SIZE
        else:
            shape = MNIST_IMG_SIZE, MNIST_IMG_SIZE
        path = path.replace('\\', '/')
        if not path.endswith('/'):
            path += '/'
        if train:
            path += 'train/'
        else:
            path += 'test/'
        labels = None
        with open(path+"labels.txt") as fin:
            labels = [generate_y_label(int(line)) for line in fin.readlines()]
        img_arrays = []
        for i in tqdm(range(min(len(labels), size)), desc='loading images'):
            img_path = path+f'{i}.jpg'
            # RGB -> Gray (to cut down calculation on rotation)
            img_arr = cv2.imread(img_path)[:, :, :1]
            if not strengthen:
                img_arr.shape = shape
                img_arr = np.array(img_arr, dtype=float) / 255.0
                img_arrays.append(img_arr)
            else:
                for _ in range(strengthenCount):
                    matrix = cv2.getRotationMatrix2D(center=(randint(5, MNIST_IMG_SIZE-5-1), randint(5, MNIST_IMG_SIZE-5-1)), 
                        angle=float(randint(-25, 25)), scale=randint(9, 11)/10.0)
                    res = cv2.warpAffine(
                        img_arr, matrix, (MNIST_IMG_SIZE, MNIST_IMG_SIZE))
                    res = np.array(res, dtype=float) / 255.0
                    res.shape = shape
                    img_arrays.append(res)
                img_arr.shape = shape
                img_arr = np.array(img_arr, dtype=float) / 255.0
                img_arrays.append(img_arr)
        if strengthen:
            temp = []
            for label in labels:
                for _ in range(1+strengthenCount):
                    temp.append(label)
            labels = temp
        if(len(img_arrays) > len(labels)):
            img_arrays = img_arrays[:len(labels)]
        elif(len(labels) > len(img_arrays)):
            labels = labels[:len(img_arrays)]
        self.dataset = ListDataset(list(zip(img_arrays, labels)))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        return self.dataset[index]
