import sys

sys.path.append('..')
import argparse
import os
import time
import random
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import get_duration, set_seed

from loss import MAE, ClsLoss
from dataset import TrajDatasetFromCache_ALL, traj_dataloader
from model import DSETA_ALL


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="")
    parser.add_argument("--n-epochs", type=int, default=50, help="")
    parser.add_argument("--batch-size", type=int, default=256, help="")
    parser.add_argument("--lr", type=float, default=0.0001, help="")
    parser.add_argument("--dp", type=float, default=0.1, help="")
    parser.add_argument("--lamda-cr", type=float, default=100., help="")
    parser.add_argument("--lamda-dspr", type=float, default=10., help="")
    parser.add_argument("--lamda-cs", type=float, default=100., help="")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--name", type=str, default="DSETA", help="")
    parser.add_argument("--sub-name", type=str, default="ALL", help="")
    parser.add_argument("--use-seed", default=False, action="store_true", help="")
    parser.add_argument("--use-writer", default=False, action="store_true", help="")
    parser.add_argument("--save-model", default=False, action="store_true", help="")
    parser.add_argument("--data-path-train", type=str, default="../data/train", help="")
    parser.add_argument("--cache-path-train", type=str, default="../data_cache/train", help="")
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


def run(
        seed,
        n_epochs,
        batch_size,
        lr,
        dp,
        lamda_cr,
        lamda_dspr,
        lamda_cs,
        device,
        name,
        sub_name,
        use_seed,
        use_writer,
        save_model,
        data_path_train,
        cache_path_train,
):
    if use_seed:
        set_seed(seed)

    device = torch.device(device)
    print(f"Device: {device}")

    model_name = f"{name}-{sub_name}_bs-{batch_size}_lr-{lr}_dp-{dp}_" \
                 f"cr-{lamda_cr}_dspr-{lamda_dspr}_cs-{lamda_cs}_sd-{seed}"
    print("Model:", model_name)

    if save_model:
        if not os.path.exists("./para"):
            os.mkdir("./para")
        model_path = f"./para/{model_name}"
        print("Model parameters are saved at:", model_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

    if use_writer:
        if not os.path.exists("./log"):
            os.mkdir("./log")
        lwriter = open(f"./log/{model_name}.train.log.csv", mode='x')
        swriter = SummaryWriter(f"./log_{name}/{sub_name}")
        # "tensorboard --logdir=log_DSETA --port=6006"
        print(f'Use "tensorboard --logdir=log_{name} --port=6006" to check the result on tensorboard')

    config = DSETA_ALL.generate_config()
    model = DSETA_ALL(*config, c_rou=12, c_seg=3, dropout=dp).to(device)  # 模型
    model.load_ds_embedding("./driver_emb_para.pkl")

    mae_train = MAE("mean").to(device)  # 损失
    cls_train = ClsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器

    file_list_train = os.listdir(data_path_train)
    lengs_train = list(map(lambda s: int(s.split(".")[0]), file_list_train))
    lengs_train.sort()

    t1 = time.time()
    for epoch in tqdm(range(n_epochs)):
        # train
        model.train()
        random.shuffle(lengs_train)
        total_loss_train = 0.
        total_loss_train_1 = 0.
        total_loss_train_2 = 0.
        total_loss_train_3 = 0.
        total_loss_train_4 = 0.
        total_loss_train_5 = 0.
        total_loss_train_6 = 0.
        num_route_train = 0
        num_seg_train = 0
        for i, leng in enumerate(lengs_train):
            dataset_train = TrajDatasetFromCache_ALL(cache_path_train, leng)
            num_route_train += len(dataset_train)
            num_seg_train += len(dataset_train) * leng
            for inputs, labels in traj_dataloader(dataset_train, batch_size):
                inputs = [x.to(device) for x in inputs]
                optimizer.zero_grad()
                pred = model(*inputs)
                out_route, out_seg, out_seg_sum = pred[0], pred[1], pred[2]
                out_ds_rou, out_dspr, out_ds_seg = pred[3], pred[4], pred[5]
                label_t, label_cum, label_seg = labels[0].to(device), labels[1].to(device), labels[2].to(device)
                label_cr, label_v, label_cs = labels[3].to(device), labels[4].to(device), labels[5].to(device)
                loss_train_1 = mae_train(out_route, label_cum)
                loss_train_2 = mae_train(out_seg_sum, label_t)
                loss_train_3 = mae_train(out_seg, label_seg)
                loss_train_eta = loss_train_1 + loss_train_2 + loss_train_3
                loss_train_4 = cls_train(out_ds_rou, label_cr)
                loss_train_5 = mae_train(out_dspr, label_v)
                loss_train_6 = cls_train(out_ds_seg, label_cs)
                loss_train_cls_rou = loss_train_4
                loss_train_dspr = loss_train_5
                loss_train_cls_seg = loss_train_6
                loss_train = loss_train_eta + lamda_cr * loss_train_cls_rou + \
                             lamda_dspr * loss_train_dspr + lamda_cs * loss_train_cls_seg
                loss_train.backward()
                optimizer.step()

                bs = labels[0].shape[0]
                loss = loss_train.cpu().item() * bs
                total_loss_train += loss
                loss_1 = loss_train_1.cpu().item() * bs * leng
                total_loss_train_1 += loss_1
                loss_2 = loss_train_2.cpu().item() * bs
                total_loss_train_2 += loss_2
                loss_3 = loss_train_3.cpu().item() * bs * leng
                total_loss_train_3 += loss_3

                loss_4 = loss_train_4.cpu().item() * bs
                total_loss_train_4 += loss_4
                loss_5 = loss_train_5.cpu().item() * bs
                total_loss_train_5 += loss_5
                loss_6 = loss_train_6.cpu().item() * bs * leng
                total_loss_train_6 += loss_6

        avg_loss_train = total_loss_train / num_route_train
        avg_loss_train_1 = total_loss_train_1 / num_seg_train
        avg_loss_train_2 = total_loss_train_2 / num_route_train
        avg_loss_train_3 = total_loss_train_3 / num_seg_train
        avg_loss_train_4 = total_loss_train_4 / num_route_train
        avg_loss_train_5 = total_loss_train_5 / num_route_train
        avg_loss_train_6 = total_loss_train_6 / num_seg_train
        t2 = get_duration(t1)
        print(f"\nEPOCH_{epoch:03d} TRAIN LOSS: {avg_loss_train:8.3f} | "
              f"REG_R : {avg_loss_train_1:8.3f} | "
              f"REG_RS: {avg_loss_train_2:8.3f} | "
              f"REG_S : {avg_loss_train_3:8.3f} |")
        print(f"                                 "
              f"CLS_R : {avg_loss_train_4:8.3f} | "
              f"DSPR_R: {avg_loss_train_5:8.3f} | "
              f"CLS_S : {avg_loss_train_6:8.3f} ({t2})")
        if use_writer:
            lwriter.write(f"T,{epoch},0,{avg_loss_train},{avg_loss_train_1},"
                          f"{avg_loss_train_2},{avg_loss_train_3},"
                          f"{avg_loss_train_4},{avg_loss_train_5},"
                          f"{avg_loss_train_6},{t2}\n")
            swriter.add_scalar(f"{model_name}: train loss", avg_loss_train, epoch)
            swriter.add_scalar(f"{model_name}: train reg r", avg_loss_train_1, epoch)
            swriter.add_scalar(f"{model_name}: train reg rs", avg_loss_train_2, epoch)
            swriter.add_scalar(f"{model_name}: train reg s", avg_loss_train_3, epoch)
            swriter.add_scalar(f"{model_name}: train cls r", avg_loss_train_4, epoch)
            swriter.add_scalar(f"{model_name}: train dspr r", avg_loss_train_5, epoch)
            swriter.add_scalar(f"{model_name}: train cls s", avg_loss_train_6, epoch)

        # save model
        if save_model:
            path = os.path.join(model_path, f"{model_name}_ep-{epoch}.pkl")
            torch.save(model.state_dict(), path)

    if use_writer:
        swriter.close()  # 关上tensorboard
        lwriter.close()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
