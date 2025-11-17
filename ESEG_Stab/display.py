import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42  
matplotlib.rcParams['ps.fonttype']  = 42
SAVE_PDF = False  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import torch
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# ============== 全局风格 ==============
sns.set_theme(
    context="notebook",
    style="whitegrid",
    rc={
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.1,
        "grid.linewidth": 0.6,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    },
)


# ============== 小工具 ==============
def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def _save(fig, path_wo_ext: str):
    try:
        fig.tight_layout()
    except Exception:
        pass

    png_path = path_wo_ext + ".png"
    fig.savefig(png_path, bbox_inches="tight")
    print(f"✅ PNG saved: {png_path}", flush=True)

    if SAVE_PDF:
        pdf_path = path_wo_ext + ".pdf"
        try:
            print(f"📝 saving PDF -> {pdf_path}", flush=True)
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"✅ PDF saved: {pdf_path}", flush=True)
        except Exception as e:
            print(f"⚠️ PDF save skipped: {e}", flush=True)

    plt.close(fig)


def _metrics(preds, targets):
    preds   = np.asarray(preds, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    m = np.isfinite(preds) & np.isfinite(targets)
    preds, targets = preds[m], targets[m]
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae  = float(mean_absolute_error(targets, preds))
    r2   = float(r2_score(targets, preds))
    pr   = float(pearsonr(targets, preds)[0]) if preds.std() > 0 and targets.std() > 0 else 0.0
    return rmse, mae, r2, pr


def _ema(x, beta=0.9):
    if x is None or len(x) == 0:
        return x
    y = []
    m = None
    for v in x:
        m = v if m is None else beta * m + (1 - beta) * v
        y.append(m)
    return y


# ============== ROC和PR曲线 ==============
def visualize_roc_pr_curves(preds, targets, save_dir, threshold=1.0):
    """
    生成ROC曲线和PR曲线
    """
    _ensure_dir(save_dir)
    
    binary_targets = (np.abs(targets) >= threshold).astype(int)
    pred_scores = np.abs(preds)  
    
    # ====== ROC曲线 ======
    fpr, tpr, _ = roc_curve(binary_targets, pred_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(fpr, tpr, color='#0072B2', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#555555', lw=1.5, linestyle='--', alpha=0.8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    
    _save(fig, os.path.join(save_dir, "roc_curve"))
    plt.close(fig)
    
    # ====== PR曲线 ======
    precision, recall, _ = precision_recall_curve(binary_targets, pred_scores)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(recall, precision, color='#D55E00', lw=2.5, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    
    _save(fig, os.path.join(save_dir, "pr_curve"))
    plt.close(fig)


# ============== 误差热力图 ==============
def visualize_error_heatmap(preds, targets, save_dir, bins=30):
    """
    生成误差热力图
    """
    _ensure_dir(save_dir)
    
    residuals = preds - targets
    
    fig, ax = plt.subforms(figsize=(7.0, 5.5))

    hb = ax.hexbin(targets, residuals, gridsize=bins, cmap='viridis', alpha=0.8)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')
    
    ax.axhline(y=0, color='#E24A33', linestyle='--', linewidth=1.5, label='Zero error')

    z = np.polyfit(targets, residuals, 1)
    p = np.poly1d(z)
    ax.plot(targets, p(targets), color='#0072B2', linewidth=2.0, label='Trend line')
    
    ax.set_xlabel('Experimental ΔΔG (kcal/mol)')
    ax.set_ylabel('Residual (Pred − Exp)')
    ax.set_title('Error Heatmap')
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    
    _save(fig, os.path.join(save_dir, "error_heatmap"))
    plt.close(fig)


# ============== 其他功能函数=============
def visualize_training_only(history, val_results, output_dir, config=None):
    """统一风格后的训练可视化"""
    _ensure_dir(output_dir)
    
    # 使用统一的训练曲线可视化
    visualize_training_curves(history, output_dir)
    
    # 生成文本报告
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report = f"""Model Performance Report
========================
Generated on: {timestamp}

[Basic Metrics]
- RMSE: {val_results['rmse']:.3f}
- MAE: {val_results.get('mae', 'NA'):.3f}
- R²: {val_results.get('r2', 'NA'):.3f}
- Pearson: {val_results.get('pearson', 'NA'):.3f}

[Training Info]
- Total Epochs: {len(history['train_loss'])}
- Best Validation Epoch: {np.argmin(history['val_rmse']) + 1 if 'val_rmse' in history else 'NA'}
"""

    if config is not None:
        report += "\n[Model Hyperparameters]\n"
        for k, v in vars(config).items():
            if k.startswith('__') or callable(v):
                continue
            report += f"- {k}: {v}\n"

    with open(os.path.join(output_dir, 'performance_report.txt'), 'w') as f:
        f.write(report)

    print(f"📊 训练阶段可视化结果图和综合报告已生成并保存到目录: {output_dir}")


def visualize_train_or_test_results(val_results, output_dir):
    """统一风格后的验证/测试结果可视化"""
    _ensure_dir(output_dir)
    
    if 'preds' in val_results and 'targets' in val_results:
        # 使用统一的预测可视化
        visualize_prediction_scatter(val_results['preds'], val_results['targets'], output_dir)
        
        # 添加ROC和PR曲线
        visualize_roc_pr_curves(val_results['preds'], val_results['targets'], output_dir)
        
        # 添加误差热力图
        visualize_error_heatmap(val_results['preds'], val_results['targets'], output_dir)
        
        print(f"✅ 验证or测试阶段分析图已生成并保存到目录: {output_dir}")


# ============== 训练曲线 ==============
def visualize_training_curves(history, save_dir, learning_rates=None, smooth=0.9):
    """
    生成：
      1) 单图版:loss_rmse_curve / val_mae_curve / val_pearson_r2
      2) 看板版:training_dashboard(四宫格含 LR)
    """
    _ensure_dir(save_dir)
    # CSV
    epochs = list(range(1, len(history["train_loss"]) + 1))
    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": history["train_loss"],
        "val_rmse": history["val_rmse"],
        "val_mae": history["val_mae"],
        "val_r2": history["val_r2"],
        "val_pearson": history["val_pearson"],
    })

    # 控制小数位
    decimals = {
        "train_loss": 4, "val_rmse": 3, "val_mae": 3, "val_r2": 3, "val_pearson": 3,
        "lr_enc": 6, "lr_out": 6,
    }
    df = df.round({k: v for k, v in decimals.items() if k in df.columns})

    if learning_rates is not None:
        # 允许 (enc_lr, out_lr) 的二元组
        if isinstance(learning_rates[0], (list, tuple)):
            df["lr_enc"] = [lr[0] for lr in learning_rates][:len(df)]
            df["lr_out"] = [lr[1] for lr in learning_rates][:len(df)]
        else:
            df["lr_enc"] = learning_rates[:len(df)]

    csv_path = os.path.join(save_dir, "train_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ 保存训练日志至: {csv_path}")

    # 可选 EMA
    train_loss = _ema(history["train_loss"], smooth) if smooth else history["train_loss"]
    val_rmse   = _ema(history["val_rmse"], smooth)   if smooth else history["val_rmse"]
    val_mae    = _ema(history["val_mae"], smooth)    if smooth else history["val_mae"]
    val_r2     = _ema(history["val_r2"], smooth)     if smooth else history["val_r2"]
    val_p      = _ema(history["val_pearson"], smooth)if smooth else history["val_pearson"]

    # ---- 单图：Loss + RMSE ----
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(train_loss, label="Train Loss", linewidth=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train Loss")
    ax2 = ax.twinx()
    ax2.plot(val_rmse, label="Val RMSE", color="#D55E00", linewidth=1.8)
    ax2.set_ylabel("Val RMSE (kcal/mol)")
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.set_title("Training Loss & Validation RMSE")
    # 合并图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    _save(fig, os.path.join(save_dir, "loss_rmse_curve"))
    plt.close(fig)

    # ---- 单图：MAE ----
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(val_mae, color="#009E73", linewidth=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("Validation MAE")
    ax.grid(True, linestyle="--", linewidth=0.6)
    _save(fig, os.path.join(save_dir, "val_mae_curve"))
    plt.close(fig)

    # ---- 单图：Pearson & R² ----
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(val_p, label="Pearson", color="#0072B2", linewidth=1.8)
    ax.plot(val_r2, label="R²", color="#CC79A7", linewidth=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Validation Pearson & R²")
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    _save(fig, os.path.join(save_dir, "val_pearson_r2"))
    plt.close(fig)

    # ---- 四宫格看板 ----
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    ax = axes[0, 0]
    ax.plot(train_loss, label="Train Loss", linewidth=1.8)
    ax2 = ax.twinx()
    ax2.plot(val_rmse, label="Val RMSE", color="#D55E00", linewidth=1.8)
    ax.set_title("Loss & RMSE"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax2.set_ylabel("RMSE")
    ax.grid(True, linestyle="--", linewidth=0.6)
    # 图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")

    ax = axes[0, 1]
    ax.plot(val_mae, color="#009E73", linewidth=1.8)
    ax.set_title("MAE"); ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (kcal/mol)")
    ax.grid(True, linestyle="--", linewidth=0.6)

    ax = axes[1, 0]
    ax.plot(val_p, label="Pearson", color="#0072B2", linewidth=1.8)
    ax.plot(val_r2, label="R²", color="#CC79A7", linewidth=1.8)
    ax.set_title("Pearson & R²"); ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, loc="best")

    ax = axes[1, 1]
    if learning_rates is not None:
        if isinstance(learning_rates[0], (list, tuple)):
            lr_enc = [lr[0] for lr in learning_rates][:len(train_loss)]
            lr_out = [lr[1] for lr in learning_rates][:len(train_loss)]
            ax.plot(lr_enc, label="LR (Encoder)", linewidth=1.6)
            ax.plot(lr_out, label="LR (Readout)", linewidth=1.6)
        else:
            ax.plot(learning_rates[:len(train_loss)], label="Learning Rate", linewidth=1.6)
        ax.set_yscale("log")
        ax.legend(frameon=False, loc="best")
    ax.set_title("Learning Rate"); ax.set_xlabel("Epoch"); ax.set_ylabel("LR (log)")
    ax.grid(True, linestyle="--", linewidth=0.6)
    for a in axes.flat:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))

    _save(fig, os.path.join(save_dir, "training_dashboard"))
    plt.close(fig)


# ============== 预测散点 + 边缘分布 + 残差 + 误差分布 ==============
def visualize_prediction_scatter(preds, targets, save_dir, title="Prediction vs. Ground Truth"):
    """
    生成三张图：
      1) scatter_ddg.png         — 回归散点
      2) residual_plot.png       — 残差图
      3) error_distribution.png  — 误差分布
    """
    _ensure_dir(save_dir)

    preds = np.asarray(preds, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    m = np.isfinite(preds) & np.isfinite(targets)
    preds, targets = preds[m], targets[m]
    rmse, mae, r2, pr = _metrics(preds, targets)
    n = preds.size

    # ====== (A) 联合散点（中心散点 + 等值线密度 + 边缘直方图）======
    df = pd.DataFrame({"True": targets, "Pred": preds}).apply(pd.to_numeric, errors="coerce").dropna()
    targets = df["True"].to_numpy(dtype=float, copy=False)
    preds   = df["Pred"].to_numpy(dtype=float, copy=False)

    low  = float(np.min([targets.min(), preds.min()]))
    high = float(np.max([targets.max(), preds.max()]))
    pad  = 0.05 * (high - low if high > low else 1.0)
    xlim = ylim = (low - pad, high + pad)

    point_c = "#4C78A8"   # 蓝
    ident_c = "#DD8452"   # 橙（对角线）
    fit_c   = "#55A868"   # 绿（拟合）

    reg = LinearRegression().fit(targets.reshape(-1, 1), preds)

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.scatter(targets, preds, s=24, alpha=0.78, linewidth=0, color=point_c, zorder=2)
    ax.plot(xlim, xlim, ls="--", color=ident_c, lw=2.1, label="Identity (y=x)", zorder=1)

    xs = np.array(xlim).reshape(-1, 1)
    ax.plot(xs, reg.predict(xs), color=fit_c, lw=2.1, label="Linear fit", zorder=3)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Experimental ΔΔG (kcal/mol)")
    ax.set_ylabel("Predicted ΔΔG (kcal/mol)")
    if title: ax.set_title("Prediction vs. Experimental")

    for side in ("top", "right"):   ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color("#666"); ax.spines[side].set_linewidth(0.9)

    txt = f"RMSE={rmse:.3f}\nMAE={mae:.3f}\nPCC={pr:.3f}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.6", lw=0.8, alpha=0.95))
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)

    _save(fig, os.path.join(save_dir, "scatter_ddg"))
    plt.close(fig)

    # ====== (B) 残差图（分箱均值 + 置信带）======
    residuals = preds - targets
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.scatter(targets, residuals, s=20, alpha=0.65, linewidth=0, color=point_c)
    ax.axhline(0, color="grey", ls="--", lw=1.2, label="Zero error")

    bins = np.linspace(targets.min(), targets.max(), 21)
    idx = np.digitize(targets, bins) - 1
    bin_x, bin_mu, bin_lo, bin_hi = [], [], [], []
    for b in range(len(bins)-1):
        msk = (idx == b)
        if msk.any():
            vals = residuals[msk]
            bin_x.append(0.5 * (bins[b] + bins[b+1]))
            mu  = vals.mean()
            sd  = vals.std(ddof=1) if vals.size > 1 else 0.0
            bin_mu.append(mu)
            bin_lo.append(mu - 1.96 * sd / max(1, np.sqrt(vals.size)))
            bin_hi.append(mu + 1.96 * sd / max(1, np.sqrt(vals.size)))
    if bin_x:
        ax.plot(bin_x, bin_mu, color=fit_c, lw=2.0, label="Binned mean")
        ax.fill_between(bin_x, bin_lo, bin_hi, color=fit_c, alpha=0.15, label="≈95% CI")

    ax.set_xlabel("Experimental ΔΔG (kcal/mol)")
    ax.set_ylabel("Residual (Pred − Exp)")
    ax.set_title("Residuals vs. Experimental ΔΔG")
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    _save(fig, os.path.join(save_dir, "residual_plot"))
    plt.close(fig)

    # ====== (C) 误差分布 ======
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bar_c    = "#1F77B4"   # 亮蓝柱
    kde_c    = "#124E8C"   # 深蓝 KDE
    median_c = "#E24A33"   # 红色虚线（中位数）

    sns.histplot(residuals, bins=30, kde=False, color=bar_c,
                edgecolor="white", linewidth=0.5, alpha=0.85, ax=ax)
    sns.kdeplot(residuals, color=kde_c, lw=2.0, ax=ax)

    mu  = residuals.mean(); med = np.median(residuals); sd = residuals.std(ddof=1)

    ax.axvline(med,     color=median_c, lw=2.2, ls="--", label=f"Median={med:.3f}")
    ax.axvline(mu,      color="#555",   lw=1.2, ls="-",  alpha=0.75, label=f"Mean={mu:.3f}")
    ax.axvline(mu - sd, color="0.65",   lw=1.0, ls=":",  alpha=0.8,  label=f"±1σ={sd:.3f}")
    ax.axvline(mu + sd, color="0.65",   lw=1.0, ls=":",  alpha=0.8)

    ax.set_xlabel("Prediction Error (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)

    _save(fig, os.path.join(save_dir, "error_distribution"))
    plt.close(fig)

    # ====== (D) 四合一报告（散点 + 残差 + 误差 + 指标框）======
    fig = plt.figure(figsize=(10.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.28)

    # 1. 散点
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(targets, preds, s=22, alpha=0.78, linewidth=0, color=point_c)
    ax.plot(xlim, xlim, ls="--", color=ident_c, lw=2.1, label="Identity (y=x)")
    xs = np.array(xlim).reshape(-1, 1)
    ax.plot(xs, reg.predict(xs), color=fit_c, lw=2.1, label="Linear fit")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Experimental ΔΔG (kcal/mol)")
    ax.set_ylabel("Predicted ΔΔG (kcal/mol)")
    ax.set_title("Prediction vs. Experimental")
    box = f"RMSE={rmse:.3f}\nMAE={mae:.3f}\nPCC={pr:.3f}"
    ax.text(0.02, 0.98, box, transform=ax.transAxes,
            va="top", ha="left", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.6", lw=0.8, alpha=0.95))
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)

    # 2. 残差
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(targets, residuals, s=20, alpha=0.65, linewidth=0, color=point_c)
    ax2.axhline(0, color="grey", ls="--", lw=1.2, label="Zero error")
    if bin_x:
        ax2.plot(bin_x, bin_mu, color=fit_c, lw=2.0, label="Binned mean")
        ax2.fill_between(bin_x, bin_lo, bin_hi, color=fit_c, alpha=0.15, label="≈95% CI")
    ax2.set_xlabel("Experimental ΔΔG (kcal/mol)")
    ax2.set_ylabel("Residual (Pred − Exp)")
    ax2.set_title("Residuals")
    ax2.legend(loc="best", frameon=True, framealpha=0.9)
    ax2.grid(alpha=0.3, linestyle="--", linewidth=0.6)

    # 3. 误差分布
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(residuals, bins=30, kde=True, color=bar_c, edgecolor=None, ax=ax3)
    ax3.axvline(mu,  color=point_c, lw=1.8, label=f"Mean={mu:.3f}")
    ax3.axvline(med, color=fit_c,   lw=1.8, ls="--", label=f"Median={med:.3f}")
    ax3.axvline(mu - sd, color="0.5", lw=1.1, ls=":", label=f"±1σ={sd:.3f}")
    ax3.axvline(mu + sd, color="0.5", lw=1.1, ls=":")
    ax3.set_xlabel("Prediction Error (kcal/mol)")
    ax3.set_ylabel("Count")
    ax3.set_title("Error Distribution")
    ax3.legend(loc="best", frameon=True, framealpha=0.9)
    ax3.grid(alpha=0.3, linestyle="--", linewidth=0.6)

    # 4. 指标信息框（整页）
    ax4 = fig.add_subplot(gs[1, 1]); ax4.axis("off")
    stats_text = (
        f"N = {n}\n\n"
        f"RMSE = {rmse:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"R² = {r2:.3f}\n"
        f"Pearson = {pr:.3f}\n"
        f"\n"
        f"Mean(True) = {targets.mean():.3f}\n"
        f"Std(True)  = {targets.std(ddof=1):.3f}\n"
        f"Mean(Pred) = {preds.mean():.3f}\n"
        f"Std(Pred)  = {preds.std(ddof=1):.3f}"
    )
    ax4.text(0.02, 0.98, stats_text, va="top", ha="left",
             fontsize=13, family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#999999", lw=1.0, alpha=0.95))

    _save(fig, os.path.join(save_dir, "prediction_report"))
    plt.close(fig)


# ============== 嵌入可视化 ==============
def visualize_embeddings(features=None, labels=None, save_dir=".", filename="embeddings.png", method="pca", **kwargs):
    full_loader = kwargs.get("full_loader", None)
    model = kwargs.get("model", None)
    device = kwargs.get("device", None)
    cls_threshold = float(kwargs.get("cls_threshold", 1.0))

    if features is None and full_loader is not None:
        feats_list, labs_list = [], []
        use_amp = (device is not None and isinstance(device, torch.device) and device.type == "cuda")
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        if model is not None and device is not None:
            model_was_train = model.training
            model.eval()

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            for batch in full_loader:
                ddg = batch.get("ddG", None)
                if ddg is not None:
                    y = ddg.detach().cpu().numpy() if torch.is_tensor(ddg) else np.asarray(ddg)
                    lab = (np.abs(y) >= cls_threshold).astype(int).reshape(-1)
                else:
                    lab = np.zeros((1,), dtype=int)

                if "features" in batch:
                    x = batch["features"]
                    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
                    x = x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)
                else:
                    x_vec = None
                    if model is not None and device is not None:
                        try:
                            wild_ids = batch["wild_ids"].to(device)
                            mut_ids  = batch["mut_ids"].to(device)
                            pos      = batch["positions"].to(device)
                            feats_in = batch["features"].to(device) if "features" in batch else None
                            with torch.no_grad():
                                if feats_in is not None:
                                    pred = model(wild_ids, mut_ids, pos, feats_in).detach()
                                else:
                                    pred = model(wild_ids, mut_ids, pos).detach()
                            x_vec = pred.view(-1, 1).cpu().numpy()
                        except Exception:
                            pass
                    if x_vec is None:

                        val = np.asarray(y if ddg is not None else lab, dtype=float).reshape(-1, 1)
                        x_vec = val
                    x = x_vec

                feats_list.append(x)
                labs_list.append(lab[: x.shape[0]])  # 对齐长度

        features = np.concatenate(feats_list, axis=0) if feats_list else None
        labels = np.concatenate(labs_list, axis=0) if labs_list else None

        if model is not None and device is not None:
            if 'model_was_train' in locals() and model_was_train:
                model.train()

    # --------- 统一到 numpy ---------
    if features is None or labels is None or len(features) == 0:
        print("⚠️ 没有可用于嵌入可视化的数据，跳过生成图。")
        return

    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    features = np.asarray(features)
    labels = np.asarray(labels).reshape(-1)
    N = features.shape[0]
    if labels.shape[0] != N:
        labels = labels[:N]

    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.shape[1] < 2:
        features = np.hstack([features, np.zeros((features.shape[0], 2 - features.shape[1]))])

    # --------- PCA 降维 ---------
    reducer = PCA(n_components=2)
    reduced = reducer.fit_transform(features)

    os.makedirs(save_dir, exist_ok=True)
    save_base = os.path.join(save_dir, os.path.splitext(filename)[0])

    # 画图
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    n_classes = len(np.unique(labels))
    show_legend = n_classes <= 20

    sns.scatterplot(
        x=reduced[:, 0], y=reduced[:, 1],
        hue=labels, palette="viridis", alpha=0.75, s=40, linewidth=0, ax=ax,
        legend=show_legend
    )
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"{method.upper()} of Sample Representations")
    ax.grid(True, linestyle="--", linewidth=0.6)
    if show_legend:
        ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., frameon=False)
    else:
        ax.legend([], [], frameon=False)

    _save(fig, save_base)
    plt.close(fig)
    print(f"✅ 保存嵌入可视化至：{save_base}.png / .pdf")


# ============== 其他功能函数 ==============
def visualize_training_only(history, val_results, output_dir, config=None):
    """统一风格后的训练可视化"""
    _ensure_dir(output_dir)
    
    # 使用统一的训练曲线可视化
    visualize_training_curves(history, output_dir)
    
    # 生成文本报告
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report = f"""Model Performance Report
========================
Generated on: {timestamp}

[Basic Metrics]
- RMSE: {val_results['rmse']:.3f}
- MAE: {val_results.get('mae', 'NA'):.3f}
- R²: {val_results.get('r2', 'NA'):.3f}
- Pearson: {val_results.get('pearson', 'NA'):.3f}

[Training Info]
- Total Epochs: {len(history['train_loss'])}
- Best Validation Epoch: {np.argmin(history['val_rmse']) + 1 if 'val_rmse' in history else 'NA'}
"""

    if config is not None:
        report += "\n[Model Hyperparameters]\n"
        for k, v in vars(config).items():
            if k.startswith('__') or callable(v):
                continue
            report += f"- {k}: {v}\n"

    with open(os.path.join(output_dir, 'performance_report.txt'), 'w') as f:
        f.write(report)

    print(f"📊 训练阶段可视化结果图和综合报告已生成并保存到目录: {output_dir}")


def visualize_train_or_test_results(val_results, output_dir):
    """统一风格后的验证/测试结果可视化"""
    _ensure_dir(output_dir)
    
    if 'preds' in val_results and 'targets' in val_results:
        # 使用统一的预测可视化
        visualize_prediction_scatter(val_results['preds'], val_results['targets'], output_dir)
        
        print(f"✅ 验证or测试阶段分析图已生成并保存到目录: {output_dir}")






