import csv
import os

import torch
import time

from tsnecuda import TSNE
import matplotlib.pyplot as plt


import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


from Utils.utils import MOUSE_10X_COLORS
from etc.global_config import config
from tqdm import tqdm


def train_on_batch(subject, num_epochs, train_iter, test_iter, optimizer, criterion, net, device, lr_jitter=False):
    algorithm = config['algorithm']
    width = []
    kan_array = ['KANformer', 'SSVEPformer3']
    if algorithm == "DDGCNN":
        lr_decay_rate = config[algorithm]['lr_decay_rate']
        optim_patience = config[algorithm]['optim_patience']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_rate,
                                                               patience=optim_patience, verbose=True, eps=1e-08)

    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_iter),
                                                               eta_min=5e-6)
    best_val_acc = 0.0

    if algorithm in kan_array:
        width = config[algorithm]['width']
        # 创建结果保存目录
        dir_path = f'../Result/classes_{config["classes"]}/{algorithm}/{str(width)}'
        print(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        # 打开csv文件
        csv_path = f'../Result/classes_{config["classes"]}/{algorithm}/{str(width)}/subject_{subject}_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]})_width({width}).csv'
    else:
        # 创建结果保存目录
        dir_path = f'../Result/classes_{config["classes"]}/{algorithm}'
        print(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        # 打开csv文件
        csv_path = f'../Result/classes_{config["classes"]}/{algorithm}/subject_{subject}_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]}).csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['subject', 'epoch', 'val_acc', 'total_params']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # 计算网络的参数量
        total_params = sum(p.numel() for p in net.parameters())
        print("网络参数量：", total_params)
        writer.writerow({'subject': subject, 'epoch': 0, 'val_acc': 0, 'total_params': total_params})
        csvfile.flush()

        # 先对FBSSVEPformer的子网络进行训练
        if algorithm == 'FBSSVEPformer':
            for i, subnetwork in enumerate(net.subnetworks):
                train_subnetwork(i, 100, train_iter, test_iter, criterion, subnetwork, device, lr_jitter)

        for epoch in range(num_epochs):
            # ==================================training procedure==========================================================
            net.train()
            sum_loss = 0.0
            sum_acc = 0.0
            progress_bar = tqdm(enumerate(train_iter), total=len(train_iter), desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch_idx, data in progress_bar:
                if algorithm == "ConvCA":
                    X, temp, y = data
                    X = X.type(torch.FloatTensor).to(device)
                    temp = temp.type(torch.FloatTensor).to(device)
                    y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    y_hat = net(X, temp)

                else:
                    X, y = data
                    X = X.type(torch.FloatTensor).to(device)
                    y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    y_hat = net(X)



                loss = criterion(y_hat, y).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if lr_jitter and algorithm != "DDGCNN":
                    scheduler.step()
                sum_loss += loss.item() / y.shape[0]
                sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()

                # Update progress bar description
                progress_bar.set_postfix({'loss': sum_loss / (batch_idx + 1), 'acc': sum_acc / (batch_idx + 1)})

            train_loss = sum_loss / len(train_iter)
            train_acc = sum_acc / len(train_iter)

            # torch.save(net, f'../Result/classes_{config["classes"]}/{algorithm}/best_Weights_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]}).pkl')
            if lr_jitter and algorithm == "DDGCNN":
                scheduler.step(train_acc)
            # print(f"epoch{epoch + 1}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}")
            # ==================================testing procedure==========================================================
            if epoch == num_epochs - 1 or (epoch + 1) % 10 == 0:
                net.eval()
                sum_acc = 0.0
                for data in test_iter:
                    if algorithm == "ConvCA":
                        X, temp, y = data
                        X = X.type(torch.FloatTensor).to(device)
                        temp = temp.type(torch.FloatTensor).to(device)
                        y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                        y_hat = net(X, temp)

                    else:
                        X, y = data
                        X = X.type(torch.FloatTensor).to(device)
                        y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                        y_hat = net(X)

                    sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()
                tsne = TSNE(n_iter=1000, verbose=1, num_neighbors=64)
                tsne_results = tsne.fit_transform(X.reshape(len(X), -1))
                plt.figure(figsize=(8, 8))
                plt.scatter(
                    x=tsne_results[:, 0],
                    y=tsne_results[:, 1],
                    c=y.cpu().numpy(),
                    cmap=plt.cm.get_cmap('Paired'),
                    alpha=0.4,
                    s=0.5
                )
                plt.title('TSNE')
                plt.show()
                val_acc = sum_acc / len(test_iter)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if algorithm in kan_array:
                        torch.save(net,
                                   f'../Result/classes_{config["classes"]}/{algorithm}/{str(width)}/best_Weights_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]})_width({width}).pkl')
                    else:
                        torch.save(net,
                                   f'../Result/classes_{config["classes"]}/{algorithm}/best_Weights_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]}).pkl')
                # 只保留四位小数的精度
                writer.writerow({'subject': subject, 'epoch': epoch + 1, 'val_acc': "%.4f" % val_acc.cpu().data.item()})
                csvfile.flush()
                print(f"epoch{epoch + 1}, val_acc={val_acc:.3f}")
    print(
        f'subject_{subject} ,training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc={val_acc:.3f}')
    if algorithm in kan_array:
        torch.save(net,
                   f'../Result/classes_{config["classes"]}/{algorithm}/{str(width)}/last_Weights_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]})_width({width}).pkl')
    else:
        torch.save(net,
                   f'../Result/classes_{config["classes"]}/{algorithm}/last_Weights_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]}).pkl')
    torch.cuda.empty_cache()
    return val_acc.cpu().data


def train_subnetwork(subject, num_epochs, train_iter, test_iter, criterion, net, device, lr_jitter=False):
    algorithm = config["algorithm"]
    lr = config[algorithm]["lr"]
    wd = config[algorithm]["wd"]
    # csv_path = f'../Result/classes_{config["classes"]}/{algorithm}/subject_{subject}_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]}).csv'
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_iter),
                                                           eta_min=5e-6)
    print(f"Training network_{subject + 1}")
    for epoch in range(num_epochs):
        # ==================================training procedure==========================================================
        net.train()
        sum_loss = 0.0
        sum_acc = 0.0
        progress_bar = tqdm(enumerate(train_iter), total=len(train_iter), desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch_idx, data in progress_bar:

            X, y = data
            X = X.type(torch.FloatTensor).to(device)
            y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
            y_hat = net(X[:, :, subject, :])

            loss = criterion(y_hat, y).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_jitter:
                scheduler.step()
            sum_loss += loss.item() / y.shape[0]
            sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()

            # Update progress bar description
            progress_bar.set_postfix({'loss': sum_loss / (batch_idx + 1), 'acc': sum_acc / (batch_idx + 1)})

        train_loss = sum_loss / len(train_iter)
        train_acc = sum_acc / len(train_iter)
        # ==================================testing procedure==========================================================
        if epoch == num_epochs - 1 or (epoch + 1) % 10 == 0:
            net.eval()
            sum_acc = 0.0
            for data in test_iter:
                if algorithm == "ConvCA":
                    X, temp, y = data
                    X = X.type(torch.FloatTensor).to(device)
                    temp = temp.type(torch.FloatTensor).to(device)
                    y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    y_hat = net(X, temp)

                else:
                    X, y = data
                    X = X.type(torch.FloatTensor).to(device)
                    y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    y_hat = net(X[:, :, subject, :])

                sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()

            val_acc = sum_acc / len(test_iter)

            print(f"epoch{epoch + 1}, val_acc={val_acc:.3f}")
    print(
        f'subNetwork_{subject + 1} ,training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc={val_acc:.4f}')


