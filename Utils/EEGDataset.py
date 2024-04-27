# Designer:Pan YuDong
# Coder:God's hand
# Time:2021/10/6 22:47
import numpy as np
from torch.utils.data import Dataset
import torch
import scipy.io
from etc.global_config import config

Ns=0 # number of subjects
classes = config['classes'] # 数据集
if classes == 12:
    Ns = config["data_param_12"]['Ns']
elif classes == 40:
    Ns = config["data_param_40"]['Ns']


class getSSVEP12Inter(Dataset):
    def __init__(self, subject=1, mode="train"):
        self.Nh = 180
        self.Nc = 8
        self.Nt = 1024
        self.Nf = 12
        self.Fs = 256
        self.eeg_raw_data = self.read_EEGData()
        self.label_raw_data = self.read_EEGLabel()
        if mode == 'train':
            self.eeg_data = torch.cat(
                (self.eeg_raw_data[0:(subject - 1) * self.Nh], self.eeg_raw_data[subject * self.Nh:]), dim=0)
            self.label_data = torch.cat(
                (self.label_raw_data[0:(subject - 1) * self.Nh:, :], self.label_raw_data[subject * self.Nh:, :]), dim=0)

        if mode == 'test':
            self.eeg_data = self.eeg_raw_data[(subject - 1) * self.Nh:subject * self.Nh]
            self.label_data = self.label_raw_data[(subject - 1) * self.Nh:subject * self.Nh]

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # get the single subject data
    def get_DataSub(self, index):
        # load file into dict
        subjectfile = scipy.io.loadmat(f'../data/Dial/DataSub_{index}.mat')
        # extract numpy from dict
        samples = subjectfile['Data']
        # (num_trial, sample_point, num_trial) => (num_trial, num_channels, sample_point)
        eeg_data = samples.swapaxes(1, 2)
        eeg_data = torch.from_numpy(eeg_data.swapaxes(0, 1))
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)
        print(eeg_data.shape)
        return eeg_data

    # 所有数据拼起来
    def read_EEGData(self):
        eeg_data = self.get_DataSub(1)
        for i in range(1, Ns):
            single_subject_eeg_data = self.get_DataSub(i + 1)
            eeg_data = torch.cat((eeg_data, single_subject_eeg_data), dim=0)
        return eeg_data

    # get the single label data
    def get_DataLabel(self, index):
        # load file into dict
        labelfile = scipy.io.loadmat(f'../data/Dial/LabSub_{index}.mat')
        # extract numpy from dict
        labels = labelfile['Label']
        label_data = torch.from_numpy(labels)
        print(label_data.shape)
        return label_data - 1

    def read_EEGLabel(self):
        label_data = self.get_DataLabel(1)
        for i in range(1, Ns):
            single_subject_label_data = self.get_DataLabel(i)
            label_data = torch.cat((label_data, single_subject_label_data), dim=0)
        return label_data


class getSSVEP12Intra(Dataset):
    def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train"):
        super(getSSVEP12Intra, self).__init__()
        self.Nh = 180  # number of trials
        self.Nc = 8  # number of channels
        self.Nt = 1024  # number of time points
        self.Nf = 12  # number of target frequency
        self.Fs = 256  # Sample Frequency
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
        subjectfile = scipy.io.loadmat(f'../data/Dial/DataSub_{self.subject}.mat')
        samples = subjectfile['Data']  # (8, 1024, 180)
        eeg_data = samples.swapaxes(1, 2)  # (8, 1024, 180) -> (8, 180, 1024)
        eeg_data = torch.from_numpy(eeg_data.swapaxes(0, 1))  # (8, 180, 1024) -> (180, 8, 1024)
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)  # (180, 1, 8, 1024)
        print(eeg_data.shape)
        return eeg_data

    # get the single label data
    def get_DataLabel(self):
        labelfile = scipy.io.loadmat(f'../data/Dial/LabSub_{self.subject}.mat')
        labels = labelfile['Label']
        print(labels)
        label_data = torch.from_numpy(labels)
        print(label_data.shape)  # torch.Size([180, 1])
        return label_data - 1


# 清华跨被试数据处理
class getSSVEP40Inter(Dataset):
    def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train"):
        super(getSSVEP40Inter, self).__init__()
        self.Nh = 240  # number of trials 6 blocks x 40 trials
        self.Nc = 9  # number of channels
        self.Nt = 1500  # number of time points
        self.Nf = 40  # number of target frequency
        self.Fs = 250  # Sample Frequency
        self.eeg_raw_data = self.read_EEGData()
        self.label_raw_data = self.read_EEGLabel()
        if mode == 'train':
            self.eeg_data = torch.cat(
                (self.eeg_raw_data[0:(subject - 1) * self.Nh], self.eeg_raw_data[subject * self.Nh:]), dim=0)
            self.label_data = torch.cat(
                (self.label_raw_data[0:(subject - 1) * self.Nh:, :], self.label_raw_data[subject * self.Nh:, :]), dim=0)

        if mode == 'test':
            self.eeg_data = self.eeg_raw_data[(subject - 1) * self.Nh:subject * self.Nh]
            self.label_data = self.label_raw_data[(subject - 1) * self.Nh:subject * self.Nh]

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    def get_DataSub(self,index):
        subjectfile = scipy.io.loadmat(f'../data/tsinghua/S{index}.mat')
        samples = subjectfile['data']  # (64, 1500, 40, 6)
        # 做一个通道选择
        # O1, Oz, O2, PO3, POZ, PO4, PZ, PO5 and PO6 对应 61, 62, 63, 55, 56, 57, 48, 54, 58
        chnls = [48, 54, 55, 56, 57, 58, 61, 62, 63]
        samples = samples[chnls, :, :, :]
        # 处理格式
        data = samples.transpose((3, 2, 0, 1))  # (6, 40, 64, 1500)
        data_stack = [data[i:i + 1, :, :, :].squeeze(0) for i in range(data.shape[0])]  # (1, 40, 64, 1500)
        eeg_data = np.vstack(data_stack)  # (240,64,1500)
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)  # (240, 1, 64, 1500) # 这个第二维度什么作用？
        eeg_data = torch.from_numpy(eeg_data)
        # print(eeg_data.shape)  # (trails, 1, channels, times)
        return eeg_data

    def read_EEGData(self):
        eeg_data = self.get_DataSub(1)
        for i in range(0, Ns-1):
            single_subject_eeg_data = self.get_DataSub(i + 1)
            eeg_data = torch.cat((eeg_data, single_subject_eeg_data), dim=0)
        return eeg_data

    # get the single label data
    def get_DataLabel(self, index):
        label_array = [[i] for i in np.tile(np.arange(self.Nf), 6)]  # 从 0 到 39 个trails循环 6 个block，每个元素是单个数组
        label_data = torch.tensor(label_array)
        # print(label_data.shape)
        return label_data  # 不需要和12分类一样-1，这里已经是从0开始了

    def read_EEGLabel(self):
        label_data = self.get_DataLabel(1)
        for i in range(0, Ns-1):
            single_subject_label_data = self.get_DataLabel(i)
            label_data = torch.cat((label_data, single_subject_label_data), dim=0)
        return label_data


# 基于清华数据集 实验由6个区块组成。每个区块包含40次试验对应40目标，持续5s
# 时间点要不要直接切掉休息的0.5s
class getSSVEP40Intra(Dataset):
    def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train"):
        super(getSSVEP40Intra, self).__init__()
        self.Nh = 240  # number of trials 6 blocks x 40 trials
        self.Nc = 9  # number of channels
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
        subjectfile = scipy.io.loadmat(f'../data/tsinghua/S{self.subject}.mat')
        samples = subjectfile['data']  # (64, 1500, 40, 6)
        # 做一个通道选择
        # O1, Oz, O2, PO3, POZ, PO4, PZ, PO5 and PO6 对应 61, 62, 63, 55, 56, 57, 48, 54, 58
        chnls = [48, 54, 55, 56, 57, 58, 61, 62, 63]
        samples = samples[chnls, :, :, :]
        # print(f'subject {self.subject} data shape: {samples.shape}')
        # 处理格式
        data = samples.transpose((3, 2, 0, 1))  # (6, 40, 64, 1500)
        data_stack = [data[i:i + 1, :, :, :].squeeze(0) for i in range(data.shape[0])]  # (1, 40, 64, 1500)
        eeg_data = np.vstack(data_stack)  # (240,64,1500)
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)  # (240, 1, 64, 1500) # 这个第二维度什么作用？
        eeg_data = torch.from_numpy(eeg_data)
        print(eeg_data.shape)  # (trails, 1, channels, times)
        return eeg_data

    # get the single label data
    def get_DataLabel(self):
        label_array = [[i] for i in np.tile(np.arange(self.Nf), 6)]  # 从 0 到 39 个trails循环 6 个block，每个元素是单个数组
        label_data = torch.tensor(label_array)
        print(label_data.shape)
        return label_data  # 不需要和12分类一样-1，这里已经是从0开始了
