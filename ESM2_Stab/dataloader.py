import os, json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from get_features import (
    calculate_sequence_features,
    normalize_features,
    parse_mutation,
    align_features_to_vector,
    FEATURE_COLUMNS,
)

class Seq_Data(Dataset):
    REQUIRED_COLS = ["wt_seq", "mt_seq", "Mutation", "ddG"]

    def __init__(self, path, use_extra=False, norm_json=None,
                 train_mean = None, train_std = None):
        
        assert os.path.exists(path), f"文件不存在: {path}"
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            raise AssertionError(f"只支持 .csv/.xlsx,收到: {ext}")

        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        assert not missing, f"缺少列: {missing}，要求: {self.REQUIRED_COLS}"

        df["wt_seq"] = df["wt_seq"].astype(str).str.upper().str.strip()
        df["mt_seq"] = df["mt_seq"].astype(str).str.upper().str.strip()
        df["Mutation"] = df["Mutation"].astype(str).str.upper().str.strip()
        df["ddG"] = pd.to_numeric(df["ddG"], errors="coerce")
        assert df["ddG"].notna().all(), "ddG 含有非数值/缺失"

        parsed = df["Mutation"].map(parse_mutation)
        bad = parsed.isna() | parsed.map(lambda t: any(v is None for v in t))
        assert not bad.any(), f"存在非法 Mutation，示例: {df.loc[bad, 'Mutation'].iloc[0]}"
        df["_pos"] = parsed.map(lambda t: int(t[1]))

        self.df = df.reset_index(drop=True)
        self.use_extra = bool(use_extra)
        self.feature_names = list(FEATURE_COLUMNS)
        self.feature_dim = len(self.feature_names)

        self.mean_dict, self.std_dict = None, None
        if self.use_extra:
            if norm_json is not None and os.path.exists(norm_json):
                with open(norm_json, "r") as f:
                    stat = json.load(f)
                self.mean_dict = {k: float(v) for k, v in zip(stat["feature_names"], stat["mean"])}
                self.std_dict  = {k: float(v) for k, v in zip(stat["feature_names"], stat["std"])}
            elif (train_mean is not None) and (train_std is not None):
                self.mean_dict = dict(train_mean)
                self.std_dict  = dict(train_std)
            else:
                self.mean_dict, self.std_dict = self._compute_feature_stats(self.df)


    def _compute_feature_stats(self, df: pd.DataFrame):
        feats = []
        for _, r in df.iterrows():
            wt, mt, mut = r["wt_seq"], r["mt_seq"], r["Mutation"]
            f = calculate_sequence_features(wt, mt, mut)  # dict
            vec = align_features_to_vector(f)              # np.float32, (7,)
            feats.append(vec)
        X = np.stack(feats, axis=0).astype(np.float32)     # (N, 7)
        mean = X.mean(axis=0)
        std  = X.std(axis=0, ddof=0)
        std[std < 1e-8] = 1e-8
        mean_dict = {k: float(v) for k, v in zip(self.feature_names, mean.tolist())}
        std_dict  = {k: float(v) for k, v in zip(self.feature_names, std.tolist())}
        return mean_dict, std_dict
    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wt = r["wt_seq"]
        mt = r["mt_seq"]
        mut = r["Mutation"]
        ddg = float(r["ddG"])
        pos = int(r["_pos"])  # 1-based

        item = {
            "wt_seq": wt,
            "mt_seq": mt,
            "pos": pos,                 # 1-based
            "ddg": ddg,                 # float
        }

        if self.use_extra:
            raw_feat = calculate_sequence_features(wt, mt, mut)
            if (self.mean_dict is not None) and (self.std_dict is not None):
                raw_feat = normalize_features(raw_feat, self.mean_dict, self.std_dict)
            vec = align_features_to_vector(raw_feat)       # np.float32, (7,)
            item["extra"] = torch.from_numpy(vec)         # torch.float32, (7,)

        return item


def collate_fn(batch):
    wt = [b["wt_seq"] for b in batch]
    mt = [b["mt_seq"] for b in batch]
    pos = torch.tensor([b["pos"] - 1 for b in batch], dtype=torch.long)    # (B,) 1-based -> 0-based
    ddg = torch.tensor([b["ddg"] for b in batch], dtype=torch.float32)     # (B,)

    out = {"wt_seq": wt, "mt_seq": mt, "pos": pos, "ddg": ddg}
    if "extra" in batch[0]:
        out["extra"] = torch.stack([b["extra"] for b in batch], dim=0)     # (B, 7)
    return out
