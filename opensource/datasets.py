import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

class TripletXRay(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, xray, index):
        self.xray = xray
        self.train = self.xray.imgs
        self.transform = self.xray.transform
        self._labels = [x[1] for x in self.train]
        self._data = [x[0] for x in self.train]
        self.labels_set = set(self._labels)
        self.label_to_indices = {label: index
                                 for index, label in enumerate(self._labels)}
        if index != 0:
            # self._labels = self.xray._labels
            # self._data = self.xray._data
            # generate fixed triplets for testing
            # self.labels_set = set(self._labels)
            # self.label_to_indices = {label: np.where(self._labels == label)[0]
            #                          for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[self._data[i],
                         self._data[np.random.choice(self.label_to_indices[self._labels[i]])],
                         self._data[np.random.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self._labels[i]]))
                                                 )
                                             ])]
                         ]
                        for i in range(len(self._data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self._data[index], self._labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self._data[positive_index]
            img3 = self._data[negative_index]
        else:
            img1 = self._data[self.test_triplets[index][0]]
            img2 = self._data[self.test_triplets[index][1]]
            img3 = self._data[self.test_triplets[index][2]]

        img1 = Image.fromarray(np.asarray(Image.open(img1)))
        img2 = Image.fromarray(np.asarray(Image.open(img2)))
        img3 = Image.fromarray(np.asarray(Image.open(img3)))
        # img1 = Image.fromarray(img1.numpy(), mode='L')
        # img2 = Image.fromarray(img2.numpy(), mode='L')
        # img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.xray)

