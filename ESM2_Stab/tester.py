# test_s669.py
import os
import time
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import Seq_Data, collate_fn
from model import ESM2_DDG_Predictor
from display import visualize_train_or_test_results, visualize_embeddings

warnings.filterwarnings("ignore")

# ----------------- Utils -----------------
def set_seed(seed=42, deterministic=True):  
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device):
    model.eval()
    total = 0
    mse_sum = 0.0
    mae_sum = 0.0
    preds_all, targets_all = [], []

    for batch in loader:
        wt_seq = batch["wt_seq"]
        mt_seq = batch["mt_seq"]

        pos0 = ((batch["pos"] - 1) if (batch["pos"].min().item() >= 1) else batch["pos"]).to(device).long()

        extra = batch.get("extra", None)
        if extra is not None:
            extra = extra.to(device)

        y = batch["ddg"].to(device)

        out = model(wt_seq, mt_seq, pos0, extra=extra)  # 单向预测 (wt -> mt)
        diff = out - y
        mse_sum += torch.sum(diff * diff).item()
        mae_sum += torch.sum(torch.abs(diff)).item()
        total += y.numel()

        preds_all.append(out.detach().cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds_all, axis=0) if preds_all else np.array([])
    targets = np.concatenate(targets_all, axis=0) if targets_all else np.array([])

    rmse = float(np.sqrt(mse_sum / max(total, 1)))
    mae = float(mae_sum / max(total, 1))

    try:
        ss_res = float(np.sum((preds - targets) ** 2))
        ss_tot = float(np.sum((targets - targets.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
    except Exception:
        r2 = 0.0

    try:
        if targets.size and np.std(targets) > 1e-8 and np.std(preds) > 1e-8:
            pearson = float(np.corrcoef(targets, preds)[0, 1])
        else:
            pearson = 0.0
    except Exception:
        pearson = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "preds": preds,
        "targets": targets,
    }


def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["model_state_dict"]
    cfg_dict   = ckpt["config"]

    use_extra = cfg_dict.get("use_extra", True)
    extra_dim = 7 if use_extra else 0   

    model = ESM2_DDG_Predictor(
        device=device,
        tune_layers=cfg_dict.get("tune_layers", 1),
        verbose=cfg_dict.get("verbose", False),
        extra_dim=extra_dim,
        dropout=cfg_dict.get("dropout", 0.25),
        radii=tuple(cfg_dict.get("radii", (8,))),
        use_center=cfg_dict.get("use_center", True),
        use_mean=cfg_dict.get("use_mean", True),
        use_max=cfg_dict.get("use_max", False),
        use_std=cfg_dict.get("use_std", True),
        use_gauss=cfg_dict.get("use_gauss", False),
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"✅ 从 checkpoint 加载模型成功，来自 epoch={ckpt.get('epoch', '?')}")
    return model, cfg_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--test_xlsx", default="datasets/S669/S669.xlsx" )
    parser.add_argument( "--train_xlsx", default="datasets/S2648/S2648_train.xlsx")
    parser.add_argument( "--checkpoint", default="cache_dir/ESM2/cheakpoints/best_val_rmse.pt" )
    parser.add_argument( "--output_dir", default="cache_dir/ESM2/output_1/testing")
    parser.add_argument( "--device", default="cuda:1")
    parser.add_argument(  "--batch_size", type=int, default=1 )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(42)
    device = torch.device(args.device)
    print("使用设备:", device)

    model, cfg_from_ckpt = load_model_from_ckpt(args.checkpoint, device)

    use_extra = True
    if cfg_from_ckpt is not None and getattr(cfg_from_ckpt, "use_extra", None) is not None:
        use_extra = bool(cfg_from_ckpt.use_extra)

    print(f"使用额外特征(use_extra) = {use_extra}")

    print("📦 加载训练集（用于 extra 归一化）:", args.train_xlsx)
    train_ds = Seq_Data(args.train_xlsx, use_extra=use_extra)

    train_mean = getattr(train_ds, "mean_dict", None)
    train_std = getattr(train_ds, "std_dict", None)

    print("📦 加载测试集 S669:", args.test_xlsx)
    test_ds = Seq_Data(
        args.test_xlsx,
        use_extra=use_extra,
        train_mean=train_mean,
        train_std=train_std,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # ===== 3. 评估 =====
    t0 = time.time()
    test_metrics = evaluate_model(model, test_loader, device)
    sec = time.time() - t0

    print("\n===== S669 测试结果 =====")
    print(f"RMSE    = {test_metrics['rmse']:.4f}")
    print(f"MAE     = {test_metrics['mae']:.4f}")
    print(f"R2      = {test_metrics['r2']:.4f}")
    print(f"Pearson = {test_metrics['pearson']:.4f}")
    print(f"耗时    = {sec:.2f} 秒")

    preds = test_metrics["preds"]
    targets = test_metrics["targets"]

    df_out = pd.DataFrame(
        {
            "ddg_true": targets,
            "ddg_pred": preds,
        }
    )
    csv_path = os.path.join(args.output_dir, "S669_predictions.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"✅ 预测结果已保存到: {csv_path}")

    # ===== 5. 可视化：散点图等 =====
    visualize_train_or_test_results(test_metrics, args.output_dir)
    try:
        visualize_embeddings(
            None,
            None,
            args.output_dir,
            "S669_embeddings.png",
            full_loader=test_loader,
            model=model,
            device=device,
        )
    except Exception as e:
        print("⚠️ 可视化 embedding 时出错（可以忽略）:", repr(e))

    print("✅ S669 测试完成，所有输出已写入:", args.output_dir)


if __name__ == "__main__":
    main()

