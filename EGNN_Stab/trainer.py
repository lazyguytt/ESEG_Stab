import os, time, random, warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from model import EGNNStab
from protein_dataset import ProteinDataset
from display import (
    visualize_training_only,
    visualize_train_or_test_results,
    visualize_embeddings
)

warnings.filterwarnings("ignore")


# =========================Config=======================================
class Config:
    def __init__(self):
        self.wild_dir   = "datasets/S2648/ESMFold/wild_pdb"
        self.mutant_dir = "datasets/S2648/ESMFold/mut_pdb"
        self.train_csv  = "datasets/S2648/S2648_train.xlsx"
        self.val_csv    = "datasets/S2648/S2648_val.xlsx"

        self.cache_dir      = "cache_dir/EGNN"
        self.output_dir     = "cache_dir/EGNN/output_1/training"
        self.checkpoint_dir = "cache_dir/EGNN/checkpoints"
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_rmse_model.pth")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # ----- Training params -----
        self.batch_size   = 8
        self.accum_steps  = 4         

        self.max_epochs   = 60

        self.lr           = 1e-4
        self.weight_decay = 5e-5
        self.grad_clip    = 2.0

        # loss
        self.huber_delta    = 1.0
        self.antisym_lambda = 0.02

        # ----- Model capacity / regularization -----
        self.depth   = 3
        self.hidden  = 256
        self.nhead   = 6
        self.dropout = 0.3

        self.normalize = True
        self.norm_node_idx = (
            list(range(20, 26)) +
            [26, 27] +
            list(range(36, 40)) +
            [40, 41, 42]
        )
        self.norm_edge_idx = [0, 41, 42]

        self.drop_hbond = False
        self.hbond_channels = None

        self.num_workers = 4
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"


# ================================================================
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================== Evaluation =====================
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
        pearson = float(np.corrcoef(targets, preds)[0, 1]) if (targets.size and np.std(targets)>1e-8 and np.std(preds)>1e-8) else 0.0
    except Exception:
        pearson = 0.0

    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson,
            "preds": preds, "targets": targets}


# ========================Training=================================
def train(cfg: Config):
    set_seed(42)
    device = torch.device(cfg.device)
    print("▶ Device:", device)

    # ---- Dataset ----
    ds_train = ProteinDataset(
        wild_dir=cfg.wild_dir, mutant_dir=cfg.mutant_dir,
        disk_cache_dir=cfg.cache_dir, ddg_csv=cfg.train_csv,
        normalize=cfg.normalize,
        norm_node_idx=cfg.norm_node_idx, norm_edge_idx=cfg.norm_edge_idx,
        cache_tag="egnn_v1", include_edge_hbond=True,
        local_density_radius=8.0,
    )

    ds_val = ProteinDataset(
        wild_dir=cfg.wild_dir, mutant_dir=cfg.mutant_dir,
        disk_cache_dir=cfg.cache_dir, ddg_csv=cfg.val_csv,
        normalize=cfg.normalize,
        norm_node_idx=cfg.norm_node_idx, norm_edge_idx=cfg.norm_edge_idx,
        cache_tag="egnn_v1", include_edge_hbond=True,
        local_density_radius=8.0, stats_mode="load-only"
    )

    # y_train = np.array([s[2] for s in ds_train.samples])
    # y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)
    # print(f"[Label stats] mean={y_mean:.4f}, std={y_std:.4f}")

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=ProteinDataset.collate_fn,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=ProteinDataset.collate_fn,
    )

    # ---- Model ----
    probe  = next(iter(train_loader))
    d_node = probe["wt"]["node_feats"].shape[1]
    d_edge = probe["wt"]["edge_attr"].shape[1]

    print(f"EGNN Input dims: d_node={d_node}, d_edge={d_edge}")

    model = EGNNStab(
        d_node=d_node,
        d_edge=d_edge,
        depth=cfg.depth,
        d_hidden=cfg.hidden,
        nhead_local=cfg.nhead,
        dropout=cfg.dropout,
        drop_hbond=cfg.drop_hbond,
        hbond_channels=cfg.hbond_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, eta_min=1e-6 )
    scaler = GradScaler(enabled=torch.cuda.is_available())  

    # ---- Log ----
    best_rmse = float("inf")
    history = {k: [] for k in ["train_loss", "val_rmse", "val_mae", "val_r2", "val_pearson"]}
    print("epoch   train_loss   val_rmse   val_mae    val_r2   val_pearson   time(s)")

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()

        running = 0.0
        seen    = 0
        optimizer.zero_grad(set_to_none=True)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.max_epochs}", leave=False)
        ):
            wt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["wt"].items()}
            mt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["mt"].items()}
            y  = batch["ddg"].to(device).float()  

            with autocast(device_type="cuda", dtype=amp_dtype):

                out_dir = model(wt, mt).squeeze(-1)
                out_rev = model(mt, wt).squeeze(-1)
                loss_data = 0.5 * (
                    nn.functional.huber_loss(out_dir, y, delta=1.0) +
                    nn.functional.huber_loss(out_rev, -y, delta=1.0)
                )
                loss_as = torch.mean((out_dir + out_rev) ** 2)
                full_loss = loss_data + cfg.antisym_lambda * loss_as
                loss = full_loss / cfg.accum_steps

            scaler.scale(loss).backward()

            running += loss.item() * y.size(0)
            seen    += y.size(0)

            if (step + 1) % cfg.accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # ---- Validation ----
        avg_train_loss = running / max(seen, 1)
        history["train_loss"].append(avg_train_loss)

        val_m = evaluate(model, val_loader, device)
        history["val_rmse"].append(val_m["rmse"])
        history["val_mae"].append(val_m["mae"])
        history["val_r2"].append(val_m["r2"])
        history["val_pearson"].append(val_m["pearson"])

        print(
            f"{epoch:5d}  "
            f"{avg_train_loss:11.4f}  "
            f"{val_m['rmse']:9.4f}  "
            f"{val_m['mae']:9.4f}  "
            f"{val_m['r2']:9.4f}  "
            f"{val_m['pearson']:13.4f}  "
            f"{time.time()-t0:9.2f}"
        )

        # ---- Save Best ----
        if val_m["rmse"] < best_rmse - 1e-12:
            best_rmse = val_m["rmse"]
            torch.save(
                {
                    "epoch": epoch,
                    "best_rmse": best_rmse,
                    "model_state_dict": model.state_dict(),
                },
                cfg.best_model_path,
            )
            print(f"✔ New best model saved @ epoch {epoch} (rmse={best_rmse:.4f})")

    # ===== Final Visualization =====
    final_val = evaluate(model, val_loader, device)
    visualize_training_only(history, final_val, cfg.output_dir, config=cfg)
    visualize_train_or_test_results(final_val, cfg.output_dir)
    visualize_embeddings(
        None,
        None,
        cfg.output_dir,
        "final_embedding.png",
        full_loader=train_loader,
        model=model,
        device=device,
    )

    print("Training finished. Best model:", cfg.best_model_path)



if __name__ == "__main__":
    from features import debug_print_feature_layout
    debug_print_feature_layout()

    cfg = Config()
    train(cfg)


