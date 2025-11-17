import os
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

from dataloader import Seq_Data, collate_fn
from model import ESM2_DDG_Predictor
from display import visualize_training_only, visualize_train_or_test_results, visualize_embeddings

warnings.filterwarnings("ignore")


# ----------------- Config -----------------
class Config:
    def __init__(self):
        self.train_path = "datasets/S2648/S2648_train.xlsx"
        self.val_path   = "datasets/S2648/S2648_val.xlsx"
        self.output_dir = "cache_dir/ESM2/output_1/training"
        self.checkpoint_dir = "cache_dir/ESM2/cheakpoints"
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_val_rmse.pt")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.use_extra = True
        self.use_center = True
        self.use_mean   = True
        self.use_max    = False
        self.use_std    = True
        self.use_gauss  = False
        
        self.verbose = False

        self.seed = 42
        self.device = "cuda:1"
        self.batch_size = 1
        self.max_epochs = 30

        self.tune_layers = 1

        self.esm_lr  = 1e-5            
        self.head_lr = 5e-4              
        self.gate_lr = 1e-3 

        self.weight_decay = 0.01          
        self.head_weight_decay = 1e-4     
        self.gate_weight_decay = 0.0 
                     
        self.radii      = (8,)
        self.dropout    = 0.3
        self.max_grad_norm = 1.0

        self.grad_accum_steps = 4 


# ----------------- Utils -----------------
def set_seed(seed=42, deterministic=True):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------- 评估 ----------------
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
        extra  = batch.get("extra", None)
        if extra is not None: extra = extra.to(device)
        y      = batch["ddg"].to(device)

        out = model(wt_seq, mt_seq, pos0, extra=extra)   # (B,)
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


# ----------------- Train -----------------
def train_model(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(device)

    # === Data ===
    train_ds = Seq_Data(cfg.train_path, use_extra=cfg.use_extra)
    val_ds   = Seq_Data(cfg.val_path,   use_extra=cfg.use_extra,
                        train_mean=getattr(train_ds, "mean_dict", None),
                        train_std=getattr(train_ds,  "std_dict",  None))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                            drop_last=False, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            drop_last=False, collate_fn=collate_fn)

    # === Model ===
    extra_dim = 7 if cfg.use_extra else 0
    model = ESM2_DDG_Predictor(
        device=device,
        tune_layers=cfg.tune_layers,
        verbose=cfg.verbose,
        extra_dim=extra_dim,
        dropout=cfg.dropout,
        radii=cfg.radii,
        use_center=cfg.use_center,
        use_mean=cfg.use_mean,
        use_max=cfg.use_max,
        use_std=cfg.use_std,
        use_gauss=cfg.use_gauss,
    ).to(device)

    # === Optimizer（三组：ESM / head_other / gate） ===
    esm_params, head_other, gate_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "esm_model" in n:
            esm_params.append(p)
        elif any(k in n for k in ["branch_logits", "layer_weights", "gamma"]):
            gate_params.append(p)
        else:
            head_other.append(p)

    opt_esm  = AdamW(esm_params,  lr=cfg.esm_lr,  weight_decay=cfg.weight_decay)
    opt_head = AdamW([
        {"params": head_other,  "lr": cfg.head_lr, "weight_decay": cfg.head_weight_decay},
        {"params": gate_params, "lr": cfg.gate_lr, "weight_decay": cfg.gate_weight_decay},
    ])

    sch_esm  = CosineAnnealingLR(opt_esm,  T_max=cfg.max_epochs, eta_min=1e-6)
    sch_head = CosineAnnealingWarmRestarts(opt_head, T_0=10, T_mult=2, eta_min=1e-6)

    scaler = GradScaler(enabled=torch.cuda.is_available())

    history = {"train_loss": [], "val_rmse": [], "val_mae": [], "val_pearson": [], "val_r2": []}
    best_val_rmse = float("inf")
    best_ckpt = None

    print("epoch   train_loss  val_rmse   val_mae    val_r2  val_pearson   time(s)")

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, seen = 0.0, 0

        opt_esm.zero_grad(set_to_none=True)
        opt_head.zero_grad(set_to_none=True)

        num_batches = len(train_loader)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for step_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.max_epochs}", leave=False),
            start=1
        ):
            wt, mt = batch["wt_seq"], batch["mt_seq"]
            pos0 = ((batch["pos"] - 1) if (batch["pos"].min().item() >= 1) else batch["pos"]).to(device).long()
            extra  = batch.get("extra", None)
            if extra is not None: extra = extra.to(device)
            y      = batch["ddg"].to(device)

            with autocast(device_type="cuda", dtype=amp_dtype):
                out_dir = model(wt, mt, pos0, extra=extra)
                out_rev = model(mt, wt, pos0, extra=extra)

                loss_data = 0.5 * (
                    nn.functional.huber_loss(out_dir, y, delta=1.0) +
                    nn.functional.huber_loss(out_rev, -y, delta=1.0)
                )
                antisym_lambda = getattr(cfg, "antisym_lambda", 0.05)
                loss_as = torch.mean((out_dir + out_rev) ** 2)
                full_loss = loss_data + antisym_lambda * loss_as
                loss = full_loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            running_loss += full_loss.item() * y.size(0)
            seen += y.size(0)

            if (step_idx % cfg.grad_accum_steps == 0) or (step_idx == num_batches):
                scaler.unscale_(opt_esm)
                scaler.unscale_(opt_head)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                scaler.step(opt_esm)
                scaler.step(opt_head)
                scaler.update()

                opt_esm.zero_grad(set_to_none=True)
                opt_head.zero_grad(set_to_none=True)

        sch_esm.step()
        sch_head.step()

        # ---- Validate ----
        avg_train_loss = running_loss / max(seen, 1)
        history["train_loss"].append(avg_train_loss)

        val_metrics = evaluate_model(model, val_loader, device)
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_pearson"].append(val_metrics["pearson"])
        history["val_r2"].append(val_metrics["r2"])

        sec = time.time() - t0
        print(f"{epoch:5d}  "
              f"{avg_train_loss:10.4f}  "
              f"{val_metrics['rmse']:8.4f}  "
              f"{val_metrics['mae']:8.4f}  "
              f"{val_metrics['r2']:8.4f}  "
              f"{val_metrics['pearson']:12.4f}  "
              f"{sec:8.3f}")

        # ---- Save best by val_rmse ----
        if val_metrics["rmse"] < best_val_rmse - 1e-12:
            best_val_rmse = val_metrics["rmse"]
            best_ckpt = {
                "epoch": epoch,
                "best_rmse": float(val_metrics["rmse"]),
                "best_mae": float(val_metrics["mae"]),
                "best_pearson": float(val_metrics["pearson"]),
                "model_state_dict": model.state_dict(),
                "config": vars(cfg),
            }
            torch.save(best_ckpt, cfg.best_model_path)
            print(f"✅ 新的最佳：epoch={epoch}, RMSE={val_metrics['rmse']:.4f}")

    # ===== 训练结束：可视化 =====
    final_val = evaluate_model(model, val_loader, device)
    visualize_training_only(history, final_val, cfg.output_dir, config=cfg)
    visualize_train_or_test_results(final_val, cfg.output_dir)
    visualize_embeddings(
        None, None, cfg.output_dir, 'final_embedding.png',
        full_loader=train_loader, model=model, device=device
    )

    if best_ckpt is not None:
        print(f"最佳模型: epoch={best_ckpt['epoch']}, RMSE={best_ckpt['best_rmse']:.4f}, "
              f"MAE={best_ckpt['best_mae']:.4f}, Pearson={best_ckpt['best_pearson']:.4f}")
    print("✅ 完成：最佳已按 val_rmse 保存 ->", cfg.best_model_path)


if __name__ == "__main__":
    cfg = Config()
    train_model(cfg)
