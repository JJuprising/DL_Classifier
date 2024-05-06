import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
class getSSVEP40Intra(Dataset):
    def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train"):
        super(getSSVEP40Intra, self).__init__()
        self.Nh = 240  # number of trials 6 blocks x 40 trials
        self.Nc = 64  # number of channels
        self.Nt = 1500  # number of time points
        self.Nf = 40  # number of target frequency
        self.Fs = 250  # Sample Frequency
        self.subject = subject  # current subject
        self.eeg_data = self.get_DataSub()
        self.label_data = self.get_DataLabel()
        self.num_trial = self.Nh // self.Nf  # number of trials of each frequency
        self.train_idx = []
        self.test_idx = []
        if KFold is not None:
            fold_trial = self.num_trial // n_splits  # number of trials in each fold
            self.valid_trial_idx = [i for i in range(KFold * fold_trial, (KFold + 1) * fold_trial)]

        for i in range(0, self.Nh, self.Nh // self.Nf):
            for j in range(self.Nh // self.Nf):
                if n_splits == 2 and j == self.num_trial - 1:
                    continue  # if K = 2, discard the last trial of each category
                if KFold is not None:  # K-Fold Cross Validation
                    if j not in self.valid_trial_idx:
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)
                else:  # Split Ratio Validation
                    if j < int(self.num_trial * train_ratio):
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)

        self.eeg_data_train = self.eeg_data[self.train_idx]
        self.label_data_train = self.label_data[self.train_idx]
        self.eeg_data_test = self.eeg_data[self.test_idx]
        self.label_data_test = self.label_data[self.test_idx]

        if mode == 'train':
            self.eeg_data = self.eeg_data_train
            self.label_data = self.label_data_train
        elif mode == 'test':
            self.eeg_data = self.eeg_data_test
            self.label_data = self.label_data_test

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # get the single subject data
    def get_DataSub(self):
        subjectfile = scipy.io.loadmat(f'data/tsinghua/S{self.subject}.mat')
        samples = subjectfile['data']  # (64, 1500, 40, 6)
        # 处理格式
        data = samples.transpose((3, 2, 0, 1)) # (6, 40, 64, 1500)
        data_stack = [data[i:i + 1, :, :, :].squeeze(0) for i in range(data.shape[0])] # (1, 40, 64, 1500)
        eeg_data = np.vstack(data_stack) # (240,64,1500)
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)  # (240, 1, 64, 1500) # 这个第二维度什么作用？
        eeg_data=torch.from_numpy(eeg_data)
        print(eeg_data.shape) # (trails, 1, channels, times)
        return eeg_data

    # get the single label data
    def get_DataLabel(self):
        label_array = [[i] for i in np.tile(np.arange(self.Nf), 6)]  # 从 0 到 39 个trails循环 6 个block，每个元素是单个数组
        label_data = torch.tensor(label_array)
        print(label_data.shape)
        return label_data # 不需要和12分类一样-1，这里已经是从0开始了


file = "data/tsinghua/S1.mat"
data = scipy.io.loadmat(file)
# print(data)
type(data)
# print(data.keys())
# print(data["data"].shape)
_label = np.arange(0, data['data'].shape[2], 1)
print(_label)
# print(data.shape)
data=data["data"]
chnls = [48, 54, 55, 56, 57, 58, 61, 62, 63]
data = data[chnls, :, :, :]
data_ori=data
# print(data.shape)
data = data.transpose((3,2,0,1)) #

data_stack = [data[i:i+1,:,:,:].squeeze(0) for i in range(data.shape[0])]
print(len(data_stack))
data = np.vstack(data_stack)
# 使用列表推导式来构建数组列表
label_array = [[i] for i in np.tile(np.arange(40), 6)]  # 从 0 到 39 循环 6 次，每个元素是单个数组
# print(label_array)
label_array=torch.tensor(label_array)
print(label_array.shape)
testSubject=1
EEGData_Train = getSSVEP40Intra(subject=testSubject, train_ratio=0.2, mode='train')
EEGData_Test = getSSVEP40Intra(subject=testSubject, train_ratio=0.2, mode='test')

