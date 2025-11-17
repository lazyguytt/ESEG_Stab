import contextlib
import torch
import torch.nn as nn
import esm

class ESM2_DDG_Predictor(nn.Module):
    """
    字符串 -> ESM2 -> 多尺度窗口差分(中心/mean/max/gauss/std)
    -> gating加权到1280 -> 层权重融合 -> 回归
    """
    def __init__(self, device=None, tune_layers=3, unfreeze_mode="last_k",
                 verbose=False, extra_dim=0, dropout=0.3, radii=(8,), gauss_sigma=4.0,
                 use_center=True, use_mean=True, use_max=False, use_std=True, use_gauss=False):

        super().__init__()
        device = torch.device('cuda:1')
        self.device = device

        self.tune_layers = int(tune_layers)
        self.unfreeze_mode = unfreeze_mode
        self.verbose = bool(verbose)
        self.esm_dim = 1280
        self.extra_dim = int(extra_dim)

        # --- 载入 ESM2 ---
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_model = self.esm_model.to(self.device)
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.padding_idx = self.esm_alphabet.padding_idx
        self.cls_idx = getattr(self.esm_alphabet, "cls_idx", None)
        self.eos_idx = getattr(self.esm_alphabet, "eos_idx", None)

        self.set_unfreeze(mode=self.unfreeze_mode, k=self.tune_layers,
                          set_repr_layers=True, verbose=self.verbose)

        self.use_center = bool(use_center)
        self.radii = tuple(int(r) for r in radii)
        self.use_mean = bool(use_mean)
        self.use_max = bool(use_max)
        self.use_std = bool(use_std)
        self.use_gauss = bool(use_gauss)
        self.gauss_sigma = float(gauss_sigma)

        branch_names = []
        if self.use_center:
            branch_names.append("center")
        for r in self.radii:
            if self.use_mean:  branch_names.append(f"mean@{r}")
            if self.use_max:   branch_names.append(f"max@{r}")
            if self.use_gauss: branch_names.append(f"gauss@{r}")
            if self.use_std:   branch_names.append(f"std@{r}")
        self.branch_names = branch_names
        self.num_branches = len(branch_names)
        assert self.num_branches > 0, "至少启用一个分支(center/mean/max/std/gauss 之一)"

        self.branch_logits = nn.Parameter(torch.zeros(self.num_branches, device=self.device))
        self.layer_weights = nn.Parameter(torch.ones(len(self.repr_layers), device=self.device))
        self.gamma = nn.Parameter(torch.tensor(0.1, device=self.device))
        self.dropout = nn.Dropout(dropout)

        in_dim = self.esm_dim + (self.extra_dim if self.extra_dim > 0 else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        ).to(self.device)

        self._init_head()

    def _rebuild_branch_logits(self, N: int):
        new_param = nn.Parameter(torch.zeros(N, device=self.device))
        if hasattr(self, "branch_logits"):
            with torch.no_grad():
                n = min(N, self.branch_logits.numel())
                if n > 0:
                    new_param[:n].copy_(self.branch_logits[:n])
        self.branch_logits = new_param


    def set_unfreeze(self, mode=None, k=None, set_repr_layers=False, verbose=None):
        """
        动态设置 ESM 冻结/解冻，并（可选）同步更新 repr_layers。
        - mode: "none" | "last_k" | "full"
        - k: 用于 "last_k"/"none"；决定解冻/融合取的最后 k 层
        - set_repr_layers: 是否同时设置 self.repr_layers = [start, ..., 33]
        """
        if mode is None:
            mode = self.unfreeze_mode
        if k is None:
            k = self.tune_layers
        if verbose is None:
            verbose = self.verbose

        for p in self.esm_model.parameters():
            p.requires_grad = False

        if mode == "full":
            for p in self.esm_model.parameters():
                p.requires_grad = True
            start = 0
        elif mode == "last_k":
            k = int(k)
            for layer in self.esm_model.layers[-k:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for m in self.esm_model.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True
            start = 34 - k
        elif mode == "none":
            start = 34 - int(k)
        else:
            raise ValueError(f"Unknown unfreeze mode: {mode}")

        if set_repr_layers:
            self.repr_layers = list(range(start, 34))  # [start, ..., 33]
            if hasattr(self, "layer_weights"):
                if self.layer_weights.numel() != len(self.repr_layers):
                    self.layer_weights = nn.Parameter(
                        torch.ones(len(self.repr_layers), device=self.device)
                    )

        if verbose:
            tot = sum(p.numel() for p in self.esm_model.parameters())
            tra = sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)
            print(f"[ESM2] mode={mode} | total={tot:,} | trainable={tra:,} | fuse_layers={getattr(self, 'repr_layers', 'unchanged')}")

    def _init_head(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _encode(self, wt_seqs, mt_seqs, pos1):
        B = len(wt_seqs)
        assert B == len(mt_seqs) == pos1.size(0)
        for i, (w, m) in enumerate(zip(wt_seqs, mt_seqs)):
            assert len(w) == len(m), f"第{i}条序列长度不同: wt={len(w)} mt={len(m)}"

        wt_pairs = [(f"wt_{i}", s) for i, s in enumerate(wt_seqs)]
        mt_pairs = [(f"mt_{i}", s) for i, s in enumerate(mt_seqs)]
        _, _, wt_tok = self.batch_converter(wt_pairs)
        _, _, mt_tok = self.batch_converter(mt_pairs)
        wt_tok = wt_tok.to(self.device); mt_tok = mt_tok.to(self.device)

        pos_tok = (pos1.to(self.device).long() + 1)  # 跳过BOS
        Lw, Lm = wt_tok.size(1), mt_tok.size(1)
        assert torch.all((pos_tok > 0) & (pos_tok < Lw) & (pos_tok < Lm)), "中心位越界(含 BOS/EOS)"

        bidx = torch.arange(B, device=self.device)
        c_w, c_m = wt_tok[bidx, pos_tok], mt_tok[bidx, pos_tok]
        bad_pad = (c_w == self.padding_idx) | (c_m == self.padding_idx)
        bad_cls = (self.cls_idx is not None) & ((c_w == self.cls_idx) | (c_m == self.cls_idx))
        bad_eos = (self.eos_idx is not None) & ((c_w == self.eos_idx) | (c_m == self.eos_idx))
        assert not (bad_pad.any() or bad_cls.any() or bad_eos.any()), "中心位命中 PAD/CLS/EOS"

        return wt_tok, mt_tok, pos_tok

    def _compute_right_bound(self, tok):
        """
        tok: (B, L) —— ESM tokens
        """
        B, L = tok.shape
        dev = tok.device
        ar = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)

        if self.eos_idx is not None:
            eos_mask = (tok == self.eos_idx)
            huge = torch.full((B, L), L + 1, dtype=torch.long, device=dev)
            first_eos = torch.where(eos_mask, ar, huge).min(dim=1).values  # 无EOS则 L+1
            if self.padding_idx is not None:
                nonpad = (tok != self.padding_idx)
                last_nonpad = torch.where(nonpad, ar, torch.full_like(ar, -1)).max(dim=1).values
            else:
                last_nonpad = torch.full((B,), L - 1, dtype=torch.long, device=dev)
            rb = torch.where(first_eos == L + 1, last_nonpad, first_eos - 1)
        else:
            if self.padding_idx is not None:
                nonpad = (tok != self.padding_idx)
                rb = torch.where(nonpad, ar, torch.full_like(ar, -1)).max(dim=1).values
            else:
                rb = torch.full((B,), L - 1, dtype=torch.long, device=dev)

        return torch.clamp(rb, min=1)  # 至少跳过 BOS=0

    def _pool_once(self, rep, c, r, kind, L, pad_left=1, pad_right=None):
        if pad_right is None:
            pad_right = L - (2 if self.eos_idx is not None else 1)

        r = int(r); c = int(c)
        s = max(pad_left, c - r)
        e = min(pad_right, c + r)
        if e < s:
            s = e = max(pad_left, min(pad_right, c))  

        seg = rep[s:e+1]  # (W, D)

        if kind == "mean":
            return seg.mean(0)
        elif kind == "max":
            return seg.max(0).values
        elif kind == "std":
            return seg.var(dim=0, unbiased=False).sqrt()
        elif kind == "gauss":
            idx = torch.arange(s, e+1, device=rep.device, dtype=rep.dtype)
            w = torch.exp(-0.5 * ((idx - float(c)) / self.gauss_sigma) ** 2)
            w = w / (w.sum() + 1e-8)
            return (seg * w.unsqueeze(-1)).sum(0)
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def _layer_gated_diff(self, wt_rep, mt_rep, pos_tok, right_bound=None):
        """
        wt_rep/mt_rep: (B, L, D)
        pos_tok: (B,)
        """
        B, L, D = wt_rep.shape
        bidx = torch.arange(B, device=wt_rep.device)
        c = pos_tok.long()
        branches = []

        # 中心差分
        if self.use_center:
            wt_c = wt_rep[bidx, c, :]   # (B, D)
            mt_c = mt_rep[bidx, c, :]
            branches.append(self.gamma * (wt_c - mt_c))

        # 窗口分支
        for r in self.radii:
            for kind, flag in (("mean", self.use_mean),
                               ("max",  self.use_max),
                               ("gauss", self.use_gauss),
                               ("std",  self.use_std)):
                if not flag:
                    continue
                w_list, m_list = [], []
                for b in range(B):
                    pr = int(right_bound[b]) if right_bound is not None else None
                    w_list.append(self._pool_once(wt_rep[b], int(c[b]), r, kind, L, pad_left=1, pad_right=pr))
                    m_list.append(self._pool_once(mt_rep[b], int(c[b]), r, kind, L, pad_left=1, pad_right=pr))
                branches.append(self.gamma * (torch.stack(w_list) - torch.stack(m_list)))  # (B, D)

        stack = torch.stack(branches, dim=0)                # (Bch, B, D)
        N = stack.size(0)
        logits = self.branch_logits
        if logits.numel() < N:
            pad = logits.new_full((N - logits.numel(),), float('-inf')) # 缺的分支权重=0
            logits = torch.cat([logits, pad], dim=0)
        elif logits.numel() > N:
            logits = logits[:N]

        gates = torch.softmax(logits, dim=0).view(N, 1, 1) # (N,1,1)
        out = (stack * gates).sum(dim=0) # -> (B, D)
        return out

    def _diff_features(self, wt_tok, mt_tok, pos_tok):
        """
            ESM 前向(WT/MT) -> 每层做多分支差分 -> 层间加权融合 -> (B,1280)
        """
        ctx = (contextlib.nullcontext() if self.training else torch.no_grad())
        with ctx:
            wt_out = self.esm_model(wt_tok, repr_layers=self.repr_layers)
            mt_out = self.esm_model(mt_tok, repr_layers=self.repr_layers)

        right_bound = self._compute_right_bound(wt_tok)

        per_layer = []
        for L_id in self.repr_layers:
            wt_rep = wt_out["representations"][L_id]
            mt_rep = mt_out["representations"][L_id]
            per_layer.append(self._layer_gated_diff(wt_rep, mt_rep, pos_tok, right_bound=right_bound))  # (B,1280)

        stack = torch.stack(per_layer, dim=0)                # (N, B, 1280)
        alpha = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1)
        fused = torch.sum(alpha * stack, dim=0)              # (B, 1280)
        return self.dropout(fused)

    def forward(self, wt_seqs, mt_seqs, pos1, extra=None):
        wt_tok, mt_tok, pos_tok = self._encode(wt_seqs, mt_seqs, pos1)
        diff = self._diff_features(wt_tok, mt_tok, pos_tok) # (B,1280)
        if self.extra_dim > 0:
            assert extra is not None and extra.size(1) == self.extra_dim
            if diff.dtype != extra.dtype: 
                extra = extra.to(diff.dtype)
            x = torch.cat([diff, extra.to(self.device)], dim=1)
        else:
            x = diff
        return self.head(x.to(torch.float32)).squeeze(-1)
