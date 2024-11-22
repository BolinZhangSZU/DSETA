import sys

sys.path.append('..')
import argparse
import os
import math
from tqdm import tqdm
import torch

from loss import MAPE, MAE, MSE, AUXAcc2, AUXAcc3

from dataset import TrajDatasetFromCache_ALL, traj_dataloader
from model import DSETA_ALL


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="")
    parser.add_argument("--bs", type=int, default=256, help="")
    parser.add_argument("--lr", type=float, default=0.0001, help="")
    parser.add_argument("--dp", type=float, default=0.1, help="")
    parser.add_argument("--lamda-cr", type=float, default=100., help="")
    parser.add_argument("--lamda-dspr", type=float, default=10., help="")
    parser.add_argument("--lamda-cs", type=float, default=100., help="")
    parser.add_argument("--batch-size", type=int, default=256, help="")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--name", type=str, default="DSETA", help="")
    parser.add_argument("--sub-name", type=str, default="ALL", help="")
    parser.add_argument("--data-path-test", type=str, default="../data/test", help="")
    parser.add_argument("--cache-path-test", type=str, default="../data_cache/test", help="")
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


def run(
        seed,
        bs,
        lr,
        dp,
        lamda_cr,
        lamda_dspr,
        lamda_cs,
        batch_size,
        device,
        name,
        sub_name,
        data_path_test,
        cache_path_test,
):
    device = torch.device(device)
    print(f"Device: {device}")

    model_name = f"{name}-{sub_name}_bs-{batch_size}_lr-{lr}_dp-{dp}_" \
                 f"cr-{lamda_cr}_dspr-{lamda_dspr}_cs-{lamda_cs}_sd-{seed}"
    print("Model:", model_name)

    config = DSETA_ALL.generate_config()
    model = DSETA_ALL(*config, c_rou=12, c_seg=3, dropout=dp).to(device)
    model_path = f"./para/{model_name}/"
    model.load_state_dict(torch.load(model_path + f"{model_name}_best_reg_rs.pkl"))

    mape_test = MAPE("sum").to(device)
    mae_test = MAE("sum").to(device)
    mse_test = MSE("sum").to(device)

    auxacc2 = AUXAcc2()
    auxacc3 = AUXAcc3()

    file_list_test = os.listdir(data_path_test)
    lengs_test = list(map(lambda s: int(s.split(".")[0]), file_list_test))
    lengs_test.sort()

    total_loss_mape_seg = 0.
    total_loss_mae_seg = 0.
    total_loss_mse_seg = 0.

    n_r_right = 0
    n_r_all = 0

    n_s_right = 0
    n_s_all = 0

    num_sample_test = 0
    model.eval()
    for leng in tqdm(lengs_test):
        dataset_test = TrajDatasetFromCache_ALL(cache_path_test, leng)
        num_sample_test += len(dataset_test)
        for inputs, labels in traj_dataloader(dataset_test, batch_size, False):
            inputs = [x.to(device) for x in inputs]
            label_t = labels[0].to(device)
            label_cr, label_cs = labels[3].to(device), labels[5].to(device)
            with torch.no_grad():
                pred = model(*inputs)
                out_route, out_seg, out_seg_sum = pred[0], pred[1], pred[2]
                loss_mape_seg = mape_test(out_seg_sum, label_t)
                loss_mae_seg = mae_test(out_seg_sum, label_t)
                loss_mse_seg = mse_test(out_seg_sum, label_t)
                total_loss_mape_seg += loss_mape_seg.cpu().item()
                total_loss_mae_seg += loss_mae_seg.cpu().item()
                total_loss_mse_seg += loss_mse_seg.cpu().item()

                out_ds_rou, out_ds_seg = pred[3], pred[5].permute(0, 2, 1)

                acc, right, all = auxacc2(out_ds_rou, label_cr)
                n_r_right += right.cpu().item()
                n_r_all += all

                acc, right, all = auxacc3(out_ds_seg, label_cs)
                n_s_right += right.cpu().item()
                n_s_all += all

    avg_loss_mape_seg = total_loss_mape_seg / num_sample_test
    avg_loss_mae_seg = total_loss_mae_seg / num_sample_test
    avg_loss_mse_seg = total_loss_mse_seg / num_sample_test

    print(f"TEST MAPE: {avg_loss_mape_seg}")
    print(f"TEST MAE : {avg_loss_mae_seg}")
    print(f"TEST RMSE: {math.sqrt(avg_loss_mse_seg)}")

    print(f"TEST ACCr: {n_r_right /n_r_all}, {n_r_right}, {n_r_all}")
    print(f"TEST ACCs: {n_s_right / n_s_all}, {n_s_right}, {n_s_all}")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
