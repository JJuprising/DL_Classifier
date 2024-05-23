# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/30 17:16
import datetime
import sys

sys.path.append('../')
import torch
import Utils.EEGDataset_plus as EEGDataset
from Utils import Ploter
from Train import Classifier_Trainer, Trainer_Script
from etc.global_config import config
import torch.multiprocessing as mp

def run():
    # 1、Define parameters of eeg
    algorithm = config['algorithm']
    classes = config['classes']
    kan_array = ['KANformer', 'SSVEPformer3']
    print(f"{'*' * 20} Current Algorithm usage: {algorithm} Using Dataset {classes} classes {'*' * 20}")
    train_radio = 0.8
    '''Parameters for training procedure'''
    UD = config["train_param"]['UD']
    ratio = config["train_param"]['ratio']
    print(f"{'*' * 20} train_param: UD-{UD} train_radio-{train_radio} {'*' * 20}")
    # if ratio == 1 or ratio == 3:
    #     Kf = 5
    #     train_ratio = 1
    # elif ratio == 2:
    #     Kf = 2

    Kf = 1
    # 训练集占比
    if ratio == 1:
        train_radio = 0.8
    elif ratio == 2:
        train_radio = 0.5
    elif ratio == 3:
        train_radio = 0.2
    '''Parameters for ssvep data'''
    if classes == 12:
        ws = config["data_param_12"]["ws"]
        Ns = config["data_param_12"]['Ns']
    if classes == 40:
        ws = config["data_param_40"]["ws"]
        Ns = config["data_param_40"]['Ns']

    '''Parameters for DL-based methods'''
    epochs = config[algorithm]['epochs']
    lr_jitter = config[algorithm]['lr_jitter']

    devices = "cuda" if torch.cuda.is_available() else "cpu"

    # 进行数据加载
    if classes == 40:
        if UD == 0:
            dataloader = EEGDataset.getSSVEP40Intra(train_ratio=train_radio)
        elif UD == 1:
            dataloader = EEGDataset.getSSVEP40Inter()
    if classes == 12:
        if UD == 0:
            dataloader = EEGDataset.getSSVEP12Intra(train_ratio=train_radio)
        elif UD == 1:
            dataloader = EEGDataset.getSSVEP12Inter()

    # 2、Start Training
    final_acc_list = []

    final_test_acc_list = []

    # 在主线程加载数据
    if classes == 40:
        if UD == 0:
            EEGData_Train = [dataloader.get_subject_data(subject=i, mode='train') for i in
                             range(Ns)]
            EEGData_Test = [dataloader.get_subject_data(subject=i, mode='test') for i in
                            range(Ns)]
        elif UD == 1:
            EEGData_Train = [dataloader.get_subject_data(subject=i+1, mode='train') for i in range(Ns)]
            EEGData_Test = [dataloader.get_subject_data(subject=i+1, mode='test') for i in range(Ns)]
    if classes == 12:
        if UD == 0:
            EEGData_Train = [dataloader.get_subject_data(subject=i, mode='train') for i in
                             range(Ns)]
            EEGData_Test = [dataloader.get_subject_data(subject=i, mode='test') for i in
                            range(Ns)]
        elif UD == 1:
            EEGData_Train = [dataloader.get_subject_data(subject=i+1, mode='train') for i in range(Ns)]
            EEGData_Test = [dataloader.get_subject_data(subject=i+1, mode='test') for i in range(Ns)]

    queue = mp.Queue()
    with mp.Pool(processes=Ns) as pool:
    #     使用multiprocessing.Queue()来存放返回值,以便可以将结果按顺序取出
        for testSubject in range(1, Ns + 1):
            pool.apply_async(train_subject,
                             args=(testSubject, epochs,classes, UD, train_radio, devices, lr_jitter, EEGData_Train[testSubject-1], EEGData_Test[testSubject-1]),
                             callback=queue.put)

            # 按照顺序从队列中获取结果
        for i in range(Ns):
            test_acc = queue.get()
            final_test_acc_list.append(test_acc)
            print(f"Subject {i + 1} Test Accuracy: {test_acc:.3f}")

    final_acc_list.append(final_test_acc_list)
    # print(final_acc_list)

    if algorithm in kan_array:
        algorithm = algorithm + '/' + str(config[algorithm]['width'])
    Ploter.plot_save_Result(final_acc_list, model_name=algorithm, dataset='classes_'+str(classes), UD=UD, ratio=ratio,
                            win_size=str(ws), text=True)

    # print(final_acc_list)
    # 3、Plot Result

# 定义每个被试的训练函数
def train_subject(testSubject, epochs, classes, UD, train_radio, devices, lr_jitter, EEGData_Train, EEGData_Test):
    print(f"开始测试第{testSubject}个被试")
    # 加载数据
    # if classes == 12:
    #     if UD == 0:
    #         # testSubject是从1开始的，而dataloder是从0开始的
    #         EEGData_Train = dataloader.getSSVEP12Intra(subject=testSubject-1, train_ratio=train_radio, mode='train')
    #         EEGData_Test = dataloader.getSSVEP12Intra(subject=testSubject-1, train_ratio=train_radio, mode='test')
    #     elif UD == 1:
    #         EEGData_Train = dataloader.get_subject_data(subject=testSubject, mode='train')
    #         EEGData_Test = dataloader.get_subject_data(subject=testSubject, mode='test')
    # elif classes == 40:
    #     if UD == 0:
    #         EEGData_Train = dataloader.getSSVEP40Intra(subject=testSubject-1, train_ratio=train_radio, mode='train')
    #         EEGData_Test = dataloader.getSSVEP40Intra(subject=testSubject-1, train_ratio=train_radio, mode='test')
    #     elif UD == 1:
    #         EEGData_Train = dataloader.get_subject_data(subject=testSubject, mode='train')
    #         EEGData_Test = dataloader.get_subject_data(subject=testSubject, mode='test')
    # print(f"对象{testSubject}的训练数据大小为{len(EEGData_Train)}，测试数据大小为{len(EEGData_Test)}")
    eeg_train_dataloader, eeg_test_dataloader = Trainer_Script.data_preprocess(EEGData_Train, EEGData_Test)

    # 定义网络
    net, criterion, optimizer = Trainer_Script.build_model(devices)

    # 训练模型
    test_acc = Classifier_Trainer.train_on_batch(testSubject, epochs, 1, eeg_train_dataloader, eeg_test_dataloader, optimizer, criterion, net, devices, lr_jitter=lr_jitter)
    return test_acc
    # return 0.8


if __name__ == '__main__':
    run()
