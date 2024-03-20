import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch


class DataLoadAdni(Dataset):
    def __init__(self, choose_data,partroi,partition,fold):
        data_path = os.path.join(os.getcwd(),'data/{}'.format(choose_data),
                                 '{}_{}_{}_{}_data.npy'.format(choose_data,partroi,partition,fold))

        label_path = os.path.join(os.getcwd(),'data/{}'.format(choose_data),
                                 '{}_{}_{}_{}_label.pkl'.format(choose_data,partroi,partition,fold))

        with open(label_path, 'rb') as f:
            self.labels= pickle.load(f)
        self.datas = torch.from_numpy(np.load(data_path).astype(np.float32))
        self.labels=torch.tensor(np.array(self.labels))

    def __getitem__(self, item):
        label = self.labels[item]
        data = self.datas[item, :, :].transpose(1,0)
        return data, label

    def __len__(self):
        return len(self.labels)