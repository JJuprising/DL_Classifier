import csv
import os

import torch
import time

# from tsnecuda import TSNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import numpy as np
from sklearn.model_selection import train_test_split



# from Utils.utils import MOUSE_10X_COLORS
from etc.global_config import config
from tqdm import tqdm



def train_on_batch(subject, num_epochs, val_interval, train_iter, test_iter, optimizer, criterion, net, device, lr_jitter=False):
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
        dir_path = f'../Result/classes_{config["classes"]}/{algorithm}/{str(width)}/pic'
        print(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        # 打开csv文件
        csv_path = f'../Result/classes_{config["classes"]}/{algorithm}/{str(width)}/subject_{subject}_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]})_width({width}).csv'
    else:
        # 创建结果保存目录
        dir_path = f'../Result/classes_{config["classes"]}/{algorithm}/pic'
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

        # 用于存储每个 epoch 的训练和验证准确度及损失
        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []
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
            train_acc = (sum_acc / len(train_iter)).item()
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            # torch.save(net, f'../Result/classes_{config["classes"]}/{algorithm}/best_Weights_ws({config["data_param_12"]["ws"]}s)_UD({config["train_param"]["UD"]}).pkl')
            if lr_jitter and algorithm == "DDGCNN":
                scheduler.step(train_acc)
            # print(f"epoch{epoch + 1}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}")
            # ==================================testing procedure==========================================================
            if epoch == num_epochs - 1 or (epoch + 1) % val_interval == 0:
                net.eval()
                sum_acc = 0.0
                sum_loss = 0.0
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
                    sum_loss += criterion(y_hat, y).sum().item()
                # tsne = TSNE(n_iter=1000, verbose=1, num_neighbors=64)
                # tsnecuda参数
                n_iter = 1000
                verbose = 1
                num_neighbors = 64

                # 使用sklearn.manifold.TSNE替换tsnecuda.TSNE
                tsne = TSNE(
                    n_components=2,  # 设置降维后的维度，默认为2
                    perplexity=30.0,  # 设置困惑度，默认为30.0
                    early_exaggeration=12.0,  # 设置早期的夸大，默认为12.0
                    learning_rate=200.0,  # 设置学习率，默认为200.0
                    n_iter=n_iter,  # 设置迭代次数，对应tsnecuda的n_iter
                    metric='euclidean',  # 设置距离度量，默认为'euclidean'
                    verbose=verbose,  # 设置是否输出详细信息，对应tsnecuda的verbose
                    random_state=None,  # 设置随机数种子
                    n_jobs=-1  # 设置使用的CPU核心数量，-1表示使用所有核心
                )

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
                val_loss = (sum_loss / len(test_iter))
                val_accuracies.append(val_acc.item())
                val_losses.append(val_loss)
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
    draw_cur(subject, dir_path, train_accuracies, val_accuracies, train_losses, val_losses, num_epochs, val_interval)
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

def draw_cur(subject, dir_path, train_accuracies, val_accuracies, train_losses, val_losses, num_epochs, val_interval=1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # print(train_accuracies)
    # print(val_accuracies)
    # print(train_losses)
    # print(val_losses)
    ax1.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    ax1.plot(range(val_interval, num_epochs + 1, val_interval), val_accuracies, label='Validation Accuracy', color='blue')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Epoch vs Accuracy")
    ax1.legend()

    ax2.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='orange')
    ax2.plot(range(val_interval, num_epochs + 1, val_interval), val_losses, label='Validation Loss', color='red')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Epoch vs Loss")
    ax2.legend()

    # 保存图表到文件
    plot_path = os.path.join(dir_path, f'training_results_{subject}.png')
    plt.savefig(plot_path)
    plt.close()
