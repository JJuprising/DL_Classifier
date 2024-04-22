import logging
import os.path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from EEG_Train.encoder.AC_BiGRU import AC_BiGRU
from EEG_Train.encoder.AITST import AITST
from EEG_Train.encoder.ConvNet import SpatialTemporalConvNet
from EEG_Train.encoder.ICTFAT import ICTFAT
from EEG_Train.encoder.MSResnet import MSResnet34
from EEG_Train.encoder.SimpleMLP import SimpleMLP
from EEG_Train.encoder.resnet import resnet18, resnet50
from supervised_train import sl_train
from EEG_Train.utils.dataset import SimpleDataset
from EEG_Train.utils.utils import init, save_val_as_excel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = init()
    LOSO_path = "./datasets/EEG_LOSO"
    subjects = os.listdir(LOSO_path)

    val_acc = []
    for subject in subjects:
        args["Test"] = subject

        if args["model"] == "resnet18":
            model = resnet18()
        if args["model"] == "resnet50":
            model = resnet50()
        if args["model"] == "AITST":
            model = AITST(input_size=(1, 5), patch_size=(1, 1), num_classes=3, model_dim=512, depth=6, heads=8,
                          mlp_dim=1024,
                          dropout=0.1, emb_dropout=0.1, channels=28)
        if args["model"] == "ICTFAT":
            model = ICTFAT(image_size=(1, 5), patch_size=(1, 1), num_classes=3, dim=512, depth=6, heads=8, mlp_dim=1024,
                           dropout=0.1, emb_dropout=0.1, channels=28, hidden_dim=512, encoder_depth=4)
        if args["model"] == "ConvNet":
            model = SpatialTemporalConvNet()
        if args["model"] == "SimpleMLP":
            model = SimpleMLP()
        if args["model"] == "MSResnet":
            model = MSResnet34()
        if args["model"] == "AC_BiGRU":
            model = AC_BiGRU(256, 3)

        model.to(device)

        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args["epoch"] // 2, eta_min=1e-5)
        metrics_dict = {"acc": Accuracy(task="multiclass", num_classes=args['num_classes']).to(device)}

        dataset_path = os.path.join(LOSO_path, f"{subject}")

        train_dataset = SimpleDataset(path=dataset_path, stage="train")
        train_dataloader = DataLoader(train_dataset, batch_size=args["bs"], shuffle=True)

        val_dataset = SimpleDataset(path=dataset_path, stage="test")
        val_dataloader = DataLoader(val_dataset, batch_size=args["bs"], shuffle=False)

        dfhistory = sl_train(args, model, optimizer, scheduler, loss_fn, metrics_dict,
                             train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                             epochs=args["epoch"], patience=args["patience"], monitor="val_acc", mode="max")
        logging.info("\n" + dfhistory.to_string())
        val_acc.append(dfhistory['val_acc'].max())

    result_save_path = "result/{0}.xlsx".format(args["exp"])
    save_val_as_excel(val_acc, result_save_path)
