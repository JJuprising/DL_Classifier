import logging
import os
import sys
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from EEG_Train.utils.utils import epoch_log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StepRunner:
    def __init__(self, net, loss_fn, stage="train", metrics_dict=None, optimizer=None):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer = optimizer

    def step(self, features, labels):
        # loss
        features, labels = features.to(device), labels.to(device)
        preds = self.net(features)
        loss = self.loss_fn(preds, labels)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # metrics
        step_metrics = {self.stage + "_" + name: metric_fn(preds, labels).item()
                        for name, metric_fn in self.metrics_dict.items()}

        return loss.item(), step_metrics

    def train_step(self, features, labels):
        self.net.train()  # 训练模式, dropout层发生作用
        return self.step(features, labels)

    @torch.no_grad()
    def eval_step(self, features, labels):
        self.net.eval()  # 预测模式, dropout层不发生作用
        return self.step(features, labels)

    def __call__(self, features, labels):
        if self.stage == "train":
            return self.train_step(features, labels)
        else:
            return self.eval_step(features, labels)


class EpochRunner:
    def __init__(self, steprunner, scheduler=None):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.scheduler = scheduler

    def __call__(self, dataloader):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout)
        for i, batch in loop:
            loss, step_metrics = self.steprunner(*batch)
            step_log = dict({self.stage + "_loss": loss}, **step_metrics)
            total_loss += loss
            step += 1
            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {self.stage + "_" + name: metric_fn.compute().item()
                                 for name, metric_fn in self.steprunner.metrics_dict.items()}
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        if self.scheduler is not None:
            self.scheduler.step()
        return epoch_log


def sl_train(args, net, optimizer, scheduler, loss_fn, metrics_dict, train_dataloader, val_dataloader=None, epochs=10,
             save_path='./checkpoint', patience=5, monitor="val_loss", mode="min"):
    train_history = {}
    logging.warning("=" * 25 + " Start Supervised Training " + "=" * 25 + "\n")
    for epoch in range(1, epochs + 1):
        epoch_log("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(net=net, stage="train", loss_fn=loss_fn, metrics_dict=deepcopy(metrics_dict),
                                       optimizer=optimizer)
        train_epoch_runner = EpochRunner(train_step_runner, scheduler=scheduler)
        train_metrics = train_epoch_runner(train_dataloader)

        for name, metric in train_metrics.items():
            train_history[name] = train_history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_dataloader:
            val_step_runner = StepRunner(net=net, stage="val", loss_fn=loss_fn, metrics_dict=deepcopy(metrics_dict))
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_dataloader)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                train_history[name] = train_history.get(name, []) + [metric]

        logging.info("Train Acc: {0:.4f} Train Loss: {1:.6f}".format(train_history["train_acc"][epoch - 1],
                                                                     train_history["train_loss"][epoch - 1]))
        logging.info("Val Acc: {0:.4f} Val Loss: {1:.6f}".format(train_history["val_acc"][epoch - 1],
                                                                 train_history["val_loss"][epoch - 1]) + "\n")

        # 3，early-stopping -------------------------------------------------
        arr_scores = train_history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            save_folder = os.path.join(save_path, "{0}".format(args["exp"]))
            subject_save_parent_folder = os.path.join(save_folder, "Weights_by_Subject")
            epoch_save_parent_folder = os.path.join(save_folder, "Weights_by_Epoch")
            subject_save_folder = os.path.join(subject_save_parent_folder, "Subject_{0}".format(args["Test"]))
            os.makedirs(subject_save_folder, exist_ok=True)
            epoch_save_folder = os.path.join(epoch_save_parent_folder, "Epoch_{0}".format(epoch))
            os.makedirs(epoch_save_folder, exist_ok=True)

            subject_ckpt_name = "Epoch{0}_acc{1:.6f}.pth".format(train_history['epoch'][best_score_idx],
                                                                train_history['val_acc'][epoch - 1])
            subject_ckpt_path = os.path.join(subject_save_folder, subject_ckpt_name)
            torch.save(net.state_dict,subject_ckpt_path)

            epoch_ckpt_name = "Subject_{0}_acc{1:.6f}.pth".format(args["Test"], train_history['val_acc'][epoch - 1])
            epoch_ckpt_path = os.path.join(epoch_save_folder, epoch_ckpt_name)
            torch.save(net.state_dict, epoch_ckpt_path)

            logging.info("<<<<<< reach best {0} : {1} >>>>>>".format(monitor, arr_scores[best_score_idx]) + "\n")
        if len(arr_scores) - best_score_idx > patience:
            logging.info(
                "<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(monitor, patience) + "\n")
            break

    return pd.DataFrame(train_history)
