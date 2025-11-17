import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from model import EGNNStab
from protein_dataset import ProteinDataset
from display import visualize_train_or_test_results

def evaluate(model, loader, device):
    model.eval()
    total = 0
    mse_sum = 0.0
    mae_sum = 0.0
    preds_all, targets_all = [], []

    with torch.no_grad():
        for batch in loader:
            wt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["wt"].items()}
            mt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["mt"].items()}
            y  = batch["ddg"].to(device).float()

            seq = batch.get("seq", None)   # List[str]
            out = model(wt, mt, seq=seq).squeeze(-1)

            diff = out - y
            mse_sum += torch.sum(diff * diff).item()
            mae_sum += torch.sum(torch.abs(diff)).item()
            total   += y.numel()

            preds_all.append(out.cpu().numpy())
            targets_all.append(y.cpu().numpy())

    preds   = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)

    rmse = float(np.sqrt(mse_sum / total))
    mae  = float(mae_sum / total)
    pearson = float(np.corrcoef(targets, preds)[0, 1])
    ss_res = float(np.sum((preds - targets) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2)) + 1e-12
    r2 = 1 - ss_res / ss_tot

    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson, 
            "preds": preds, "targets": targets}

# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda:1")

    # ===== Load test dataset (S669) =====
    ds_test = ProteinDataset(
        wild_dir="datasets/S669/ESMFold/wild_pdb",
        mutant_dir="datasets/S669/ESMFold/mt_pdb",
        disk_cache_dir="cache_dir/EGNN+ESM2",
        ddg_csv="datasets/S669/S669.xlsx",
        normalize=True,
        cache_tag="egnn_v1",
        include_edge_hbond=True,
        stats_mode="load-only",
    )

    test_loader = DataLoader(
        ds_test, batch_size=8, shuffle=False,
        num_workers=4, collate_fn=ProteinDataset.collate_fn
    )

    # ===== Restore best model =====
    ckpt_path = "cache_dir/EGNN+ESM2/cheakpoints/best_rmse_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device)

    # Build model with config in checkpoint
    cfg_dict = ckpt.get("config", {})
    d_node = next(iter(test_loader))["wt"]["node_feats"].shape[1]
    d_edge = next(iter(test_loader))["wt"]["edge_attr"].shape[1]

    model = EGNNStab(
        d_node=d_node,
        d_edge=d_edge,
        depth=cfg_dict.get("depth", 3),
        d_hidden=cfg_dict.get("hidden", 256),
        nhead_local=cfg_dict.get("nhead", 6),
        dropout=cfg_dict.get("dropout", 0.3),
        drop_hbond=cfg_dict.get("drop_hbond", False),
        hbond_channels=cfg_dict.get("hbond_channels", None),
        use_esm=cfg_dict.get("use_esm", True),
        esm_model_name=cfg_dict.get("esm_model_name", "esm2_t33_650M_UR50D"),
        esm_layer=cfg_dict.get("esm_layer", -1),
        esm_proj_dim=cfg_dict.get("esm_proj_dim", 64),
        freeze_esm=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded best model from:", ckpt_path)

    # ===== Evaluate =====
    result = evaluate(model, test_loader, device)
    print("Test RMSE:   ", result["rmse"])
    print("Test MAE:    ", result["mae"])
    print("Test Pearson:", result["pearson"])
    print("Test R2:     ", result["r2"])

    # ===== Scatter plot =====
    outdir = "cache_dir/EGNN+ESM2/output_1/testing"
    os.makedirs(outdir, exist_ok=True)
    visualize_train_or_test_results(result, outdir)
    print("Scatter plot saved in", outdir)

