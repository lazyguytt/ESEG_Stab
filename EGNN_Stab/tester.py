import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import EGNNStab
from protein_dataset import ProteinDataset
from display import visualize_train_or_test_results


# ==================== Evaluation（和训练里的基本一致） ====================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    mse_sum = 0.0
    mae_sum = 0.0
    preds_all, targets_all = [], []

    for batch in loader:
        wt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["wt"].items()}
        mt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["mt"].items()}
        y  = batch["ddg"].to(device).float()

        out = model(wt, mt).squeeze(-1)

        diff = out - y
        mse_sum += torch.sum(diff * diff).item()
        mae_sum += torch.sum(torch.abs(diff)).item()
        total   += y.numel()

        preds_all.append(out.detach().cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())

    preds   = np.concatenate(preds_all, axis=0) if preds_all else np.array([])
    targets = np.concatenate(targets_all, axis=0) if targets_all else np.array([])

    rmse = float(np.sqrt(mse_sum / max(total, 1)))
    mae  = float(mae_sum / max(total, 1))

    try:
        ss_res = float(np.sum((preds - targets) ** 2))
        ss_tot = float(np.sum((targets - targets.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
    except Exception:
        r2 = 0.0
    try:
        pearson = float(np.corrcoef(targets, preds)[0, 1]) if (targets.size and np.std(targets) > 1e-8 and np.std(preds) > 1e-8) else 0.0
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


# ============================ Main ============================
if __name__ == "__main__":
    # 和训练脚本保持一致
    wild_dir   = "datasets/S669/ESMFold/wild_pdb"
    mutant_dir = "datasets/S669/ESMFold/mt_pdb"
    ddg_csv    = "datasets/S669/S669.xlsx"      # 如果你是 csv 就改成对应文件名

    cache_dir  = "cache_dir/EGNN"
    ckpt_path  = os.path.join(cache_dir, "checkpoints", "best_rmse_model.pth")
    outdir     = "cache_dir/EGNN/output_1/testing"
    os.makedirs(outdir, exist_ok=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("▶ Device:", device)

    # —— 测试集 Dataset（沿用训练时的归一化统计：stats_mode="load-only"）——
    ds_test = ProteinDataset(
        wild_dir=wild_dir,
        mutant_dir=mutant_dir,
        disk_cache_dir=cache_dir,
        ddg_csv=ddg_csv,
        normalize=True,
        norm_node_idx=(
            list(range(20, 26)) +
            [26, 27] +
            list(range(36, 40)) +
            [40, 41, 42]
        ),
        norm_edge_idx=[0, 41, 42],
        cache_tag="egnn_v1",
        include_edge_hbond=True,
        local_density_radius=8.0,
        stats_mode="load-only",   # ★ 只读取训练算好的 mean/std
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=ProteinDataset.collate_fn,
    )

    # —— 构建模型结构（和训练保持一致）——
    probe  = next(iter(test_loader))
    d_node = probe["wt"]["node_feats"].shape[1]
    d_edge = probe["wt"]["edge_attr"].shape[1]

    print(f"EGNN Input dims (test): d_node={d_node}, d_edge={d_edge}")

    model = EGNNStab(
        d_node=d_node,
        d_edge=d_edge,
        depth=3,
        d_hidden=256,
        nhead_local=6,
        dropout=0.3,
        drop_hbond=False,
        hbond_channels=None,
    ).to(device)

    # —— 加载训练时保存的最佳模型权重 —— 
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best model from: {ckpt_path}")
    print("Best RMSE in training:", ckpt.get("best_rmse", "N/A"), " (epoch maybe:", ckpt.get("epoch", "N/A"), ")")

    # —— 在 S669 上评估 —— 
    result = evaluate(model, test_loader, device)
    print("Test RMSE:   ", result["rmse"])
    print("Test MAE:    ", result["mae"])
    print("Test Pearson:", result["pearson"])
    print("Test R2:     ", result["r2"])

    # —— 画测试散点图（和训练用的函数完全一样）——
    visualize_train_or_test_results(result, outdir)
    print("Scatter plot saved in:", outdir)
