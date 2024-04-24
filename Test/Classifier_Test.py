# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/30 17:16
import datetime
import sys

sys.path.append('../')
import torch
import Utils.EEGDataset as EEGDataset
from Utils import Ploter
from Train import Classifier_Trainer, Trainer_Script
from etc.global_config import config


def run():
    # 1、Define parameters of eeg
    algorithm = config['algorithm']
    classes=config['classes']
    print(f"{'*' * 20} Current Algorithm usage: {algorithm} {'*' * 20}")

    '''Parameters for training procedure'''
    UD = config["train_param"]['UD']
    ratio = config["train_param"]['ratio']
    if ratio == 1 or ratio == 3:
        Kf = 5
    elif ratio == 2:
        Kf = 2

    Kf = 1

    '''Parameters for ssvep data'''
    if classes==12:
        ws = config["data_param_12"]["ws"]
        Ns = config["data_param_12"]['Ns']
    if classes==40:
        ws = config["data_param_40"]["ws"]
        Ns = config["data_param_40"]['Ns']

    '''Parameters for DL-based methods'''
    epochs = config[algorithm]['epochs']
    lr_jitter = config[algorithm]['lr_jitter']

    devices = "cuda" if torch.cuda.is_available() else "cpu"

    # 2、Start Training
    final_acc_list = []
    for fold_num in range(Kf):
        final_test_acc_list = []
        print(f"Training for K_Fold {fold_num + 1}")
        for testSubject in range(1, Ns + 1):
            # **************************************** #
            '''12-class SSVEP Dataset'''
            # -----------Intra-Subject Experiments--------------
            # EEGData_Train = EEGDataset.getSSVEP12Intra(subject=testSubject, KFold=fold_num, n_splits=Kf,
            #                                            mode='train')
            # EEGData_Test = EEGDataset.getSSVEP12Intra(subject=testSubject, KFold=fold_num, n_splits=Kf,
            #                                           mode='test')
            #
            # if ratio == 3:
            #     Temp = EEGData_Train
            #     EEGData_Train = EEGData_Test
            #     EEGData_Test = Temp

            # -----------12class Intra-Subject Experiments--------------
            if classes==12:
                EEGData_Train = EEGDataset.getSSVEP12Intra(subject=testSubject, train_ratio=0.8, mode='train')
                EEGData_Test = EEGDataset.getSSVEP12Intra(subject=testSubject, train_ratio=0.8, mode='test')

            # -----------12class Inter-Subject Experiments--------------
            # EEGData_Train = EEGDataset.getSSVEP12Inter(subject=testSubject, mode='train')
            # EEGData_Test = EEGDataset.getSSVEP12Inter(subject=testSubject, mode='test')

            # -----------40class Intra-Subject Experiments--------------
            if classes==40:
                EEGData_Train = EEGDataset.getSSVEP40Intra(subject=testSubject, train_ratio=0.8, mode='train')
                EEGData_Test = EEGDataset.getSSVEP40Intra(subject=testSubject, train_ratio=0.8, mode='test')

            eeg_train_dataloader, eeg_test_dataloader = Trainer_Script.data_preprocess(EEGData_Train, EEGData_Test)

            # Define Network
            net, criterion, optimizer = Trainer_Script.build_model(devices)
            test_acc = Classifier_Trainer.train_on_batch(epochs, eeg_train_dataloader, eeg_test_dataloader, optimizer,
                                                         criterion,
                                                         net, devices, lr_jitter=lr_jitter)
            final_test_acc_list.append(test_acc)
            print(f"Subject {testSubject} Test Accuracy: {test_acc:.3f}")
        final_acc_list.append(final_test_acc_list)

    # print(final_acc_list)
    # 3、Plot Result
    Ploter.plot_save_Result(final_acc_list, model_name=algorithm, dataset='DatasetA', UD=UD, ratio=ratio,
                            win_size=str(ws), text=True)


if __name__ == '__main__':
    run()
