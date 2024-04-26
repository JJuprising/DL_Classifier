import csv
import os

import torch
import time
from etc.global_config import config
from tqdm import tqdm

def train_on_batch(subject, num_epochs, train_iter, test_iter, optimizer, criterion, net, device, lr_jitter=False):
    algorithm = config['algorithm']
    if algorithm == "DDGCNN":
        lr_decay_rate = config[algorithm]['lr_decay_rate']
        optim_patience = config[algorithm]['optim_patience']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_rate,
                                                               patience=optim_patience, verbose=True, eps=1e-08)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_iter),
                                                               eta_min=5e-6)
    best_val_acc = 0.0
    # 创建结果保存目录
    dir_path = f'../Result/classes_{config["classes"]}/{algorithm}'
    print(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    # 打开csv文件
    csv_path = f'../Result/classes_{config["classes"]}/{algorithm}/subject_{subject}_log.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['subject', 'epoch', 'val_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'subject': subject, 'epoch': 0, 'val_acc': 0})
        csvfile.flush()

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
            if lr_jitter and algorithm == "DDGCNN":
                scheduler.step(train_acc)
            # print(f"epoch{epoch + 1}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}")
            # ==================================testing procedure==========================================================
            if epoch == num_epochs - 1 or (epoch+1) % 10 == 0:
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

                val_acc = sum_acc / len(test_iter)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(net.state_dict(), f'../Result/classes_{config["classes"]}/{algorithm}/best_Weights(1.0S).pkl')
                # 只保留四位小数的精度
                writer.writerow({'subject': subject, 'epoch': epoch + 1, 'val_acc': "%.4f" % val_acc.cpu().data.item()})
                csvfile.flush()
                print(f"epoch{epoch + 1}, val_acc={val_acc:.3f}")
    print(
        f'subject_{subject} ,training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc={val_acc:.3f}')
    torch.save(net.state_dict(), f'../Result/classes_{config["classes"]}/{algorithm}/last_Weights(1.0S).pkl')
    torch.cuda.empty_cache()
    return val_acc.cpu().data
