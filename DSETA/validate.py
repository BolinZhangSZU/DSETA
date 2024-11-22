import sys

sys.path.append('..')
import argparse
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import get_duration

from loss import MAE
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
    parser.add_argument("--use-writer", default=False, action="store_true", help="")
    parser.add_argument("--data-path-val", type=str, default="../data/val", help="")
    parser.add_argument("--cache-path-val", type=str, default="../data_cache/val", help="")
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
        use_writer,
        data_path_val,
        cache_path_val,
):
    device = torch.device(device)
    print(f"Device: {device}")

    model_name = f"{name}-{sub_name}_bs-{batch_size}_lr-{lr}_dp-{dp}_" \
                 f"cr-{lamda_cr}_dspr-{lamda_dspr}_cs-{lamda_cs}_sd-{seed}"
    print("Model:", model_name)

    model_path = f"./para/{model_name}"
    print("Model parameters are saved at:", model_path)

    if use_writer:
        if not os.path.exists("./log"):
            os.mkdir("./log")
        lwriter = open(f"./log/{model_name}.val.log.csv", mode='x')
        swriter = SummaryWriter(f"./log_{name}/{sub_name}")
        # 使用"tensorboard --logdir=log_DSETA --port=6006"查看tensorboard
        print(f'Use "tensorboard --logdir=log_{name} --port=6006" to check the result on tensorboard')

    config = DSETA_ALL.generate_config()
    model = DSETA_ALL(*config, c_rou=12, c_seg=3, dropout=dp).to(device)  # 模型

    mae_val = MAE("sum").to(device)  # 损失

    file_list_val = os.listdir(data_path_val)
    lengs_val = list(map(lambda s: int(s.split(".")[0]), file_list_val))
    lengs_val.sort()

    min_loss_val_1 = None
    min_loss_val_2 = None
    min_epoch_1 = None
    min_epoch_2 = None

    t1 = time.time()
    for epoch in range(n_epochs):
        model.load_state_dict(torch.load(os.path.join(model_path, f"{model_name}_ep-{epoch}.pkl")))

        # validate
        model.eval()

        total_loss_val_1 = 0.
        total_loss_val_2 = 0.

        num_route_val = 0

        for i, leng in enumerate(lengs_val):
            dataset_val = TrajDatasetFromCache_ALL(cache_path_val, leng)
            num_route_val += len(dataset_val)

            for inputs, labels in traj_dataloader(dataset_val, batch_size, False):
                inputs = [x.to(device) for x in inputs]  # 放到gpu里
                label_t = labels[0].to(device)
                with torch.no_grad():
                    pred = model(*inputs)
                    out_route, out_seg, out_seg_sum = pred[0], pred[1], pred[2]

                    out_rou = out_route[:, -1, :]
                    loss_val_1 = mae_val(out_rou, label_t)
                    loss_val_2 = mae_val(out_seg_sum, label_t)

                    loss_1 = loss_val_1.cpu().item()
                    total_loss_val_1 += loss_1
                    loss_2 = loss_val_2.cpu().item()
                    total_loss_val_2 += loss_2

        avg_loss_val_1 = total_loss_val_1 / num_route_val
        avg_loss_val_2 = total_loss_val_2 / num_route_val
        t2 = get_duration(t1)
        print(f"EPOCH_{epoch:03d} "
              f"REG_R: {avg_loss_val_1:8.3f} | "
              f"REG_RS: {avg_loss_val_2:8.3f} ({t2})")
        if use_writer:
            lwriter.write(f"{epoch},{avg_loss_val_1},{avg_loss_val_2},{t2}\n")
            swriter.add_scalar(f"{model_name}: val reg r", avg_loss_val_1, epoch)
            swriter.add_scalar(f"{model_name}: val reg rs", avg_loss_val_2, epoch)


        if min_loss_val_1 is None:
            min_loss_val_1 = avg_loss_val_1
            min_epoch_1 = epoch
            min_loss_val_2 = avg_loss_val_2
            min_epoch_2 = epoch
        else:
            if min_loss_val_1 > avg_loss_val_1:
                min_loss_val_1 = avg_loss_val_1
                min_epoch_1 = epoch
            if min_loss_val_2 > avg_loss_val_2:
                min_loss_val_2 = avg_loss_val_2
                min_epoch_2 = epoch

    print("rou:", min_epoch_1, min_loss_val_1)
    print("seg:", min_epoch_2, min_loss_val_2)

    model.load_state_dict(torch.load(os.path.join(model_path, f"{model_name}_ep-{min_epoch_1}.pkl")))
    torch.save(model.state_dict(), os.path.join(model_path, f"{model_name}_best_reg_r.pkl"))
    model.load_state_dict(torch.load(os.path.join(model_path, f"{model_name}_ep-{min_epoch_2}.pkl")))
    torch.save(model.state_dict(), os.path.join(model_path, f"{model_name}_best_reg_rs.pkl"))

    if use_writer:
        swriter.close()  # 关上tensorboard
        lwriter.close()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
