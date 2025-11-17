import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter


def mlp(din, dh, dout, n=2, act=nn.SiLU, dropout=0.2, bias_last=True):
    """
      din -> dh x (n-1) -> dout
    - 中间层: Linear + 激活 (+ 可选 Dropout)
    """
    layers = []
    dims = [din] + [dh] * (n - 1) + [dout]

    for i in range(len(dims) - 1):
        in_dim, out_dim = dims[i], dims[i + 1]
        use_bias = (i < len(dims) - 2) or bias_last
        layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
        if i < len(dims) - 2:
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


class ZeroInitLazyLinear(nn.LazyLinear):
    """ Lazy Linear,权重和偏置初始化为 0 """
    def reset_parameters(self):
        super().reset_parameters()
        if not isinstance(self.weight, UninitializedParameter):
            nn.init.zeros_(self.weight)
        if getattr(self, "bias", None) is not None and not isinstance(self.bias, UninitializedParameter):
            nn.init.zeros_(self.bias)


class EGNNLayer(nn.Module):
    """
    EGNN 单层: 消息传递 -> 节点特征更新 -> 坐标更新
    """
    def __init__(self, d_node, d_edge, d_hidden, update_coords=True,
                 coord_step=0.1, dropout=0.0, res_scale=0.3):
        super().__init__()
        self.update_coords = bool(update_coords)
        self.coord_step = float(coord_step)
        self.res_scale = float(res_scale)

        self.phi_m = mlp(d_node * 2 + d_edge, d_hidden, d_hidden, n=2, dropout=dropout)
        self.phi_h = mlp(d_hidden + d_node, d_hidden, d_node, n=2, dropout=dropout)
        self.phi_x = mlp(d_hidden, d_hidden, 1, n=2, dropout=dropout)

        self.norm_h = nn.LayerNorm(d_node)
        self.norm_m = nn.LayerNorm(d_hidden)
        self.msg_dropout = nn.Dropout(dropout)

    def forward(self, x, h, edge_index, edge_attr):

        assert x.dim() == 2 and x.size(1) == 3, f"x shape={tuple(x.shape)}"
        assert h.dim() == 2, f"h shape={tuple(h.shape)}"
        assert edge_index.dim() == 2 and edge_index.size(0) == 2 \
            and edge_index.dtype == torch.long, f"edge_index shape={tuple(edge_index.shape)}"
        assert edge_attr.dim() == 2 and edge_attr.size(0) == edge_index.size(1), \
            f"edge_attr shape={tuple(edge_attr.shape)}, E={edge_index.size(1)}"

        dev, dt = h.device, h.dtype
        x = x.to(dev, dt)
        h = h.to(dev, dt)
        edge_index = edge_index.to(dev, torch.long)
        edge_attr = edge_attr.to(dev, dt)

        if edge_index.numel() == 0:
            return x, h

        N = h.size(0)
        u, v = edge_index           # u -> v
        diff = x[u] - x[v]                           # (E,3)
        dist = diff.norm(dim=-1).clamp_min(1e-8)     # (E,)

        msg_in = torch.cat([h[u], h[v], edge_attr], dim=-1)  # (E, 2F + D)
        m_ij = self.phi_m(msg_in)                            # (E, d_hidden)
        m_ij = self.msg_dropout(m_ij)

        m_sum = h.new_zeros(N, m_ij.size(-1))
        m_sum.index_add_(0, u, m_ij.to(m_sum.dtype))

        deg = h.new_zeros(N, 1)
        deg.index_add_(0, u, deg.new_ones((u.numel(), 1)))
        m_i = m_sum / deg.clamp_min(1.0)

        h_in = self.norm_h(h)
        m_i = self.norm_m(m_i)
        h = h + self.res_scale * self.phi_h(torch.cat([h_in, m_i], dim=-1))

        if self.update_coords:
            dir_ij = diff / dist.unsqueeze(-1)                 # (E,3)
            w_ij = torch.tanh(self.phi_x(m_ij)) * self.coord_step  # (E,1)

            dx = x.new_zeros(x.size(0), 3)
            step_uv = (w_ij * dir_ij)

            dx.index_add_(0, u, step_uv.to(dx.dtype))
            dx.index_add_(0, v, (-step_uv).to(dx.dtype))

            deg_x = x.new_zeros(x.size(0), 1)
            one = deg_x.new_ones(w_ij.size())
            deg_x.index_add_(0, u, one)
            deg_x.index_add_(0, v, one)

            x = x + dx / deg_x.clamp_min(1.0)

        return x, h


class CenterGeoAttention(nn.Module):
    """
    Center + Global Geo-aware Attention
    - 使用 rbf_ic / seqsep_ic + local_bias 作为几何/序列偏置
    - 注意力仍在全图上做 softmax，避免过度丢失全局信息
    """

    def __init__(self, d_model, nhead=4, bias_dim=0,
                 attn_dropout=0.2, head_dropout=0.1, res_scale=0.5):
        
        super().__init__()
        if d_model % nhead != 0:
            g = math.gcd(d_model, nhead)
            nhead_new = max(1, g)
            print(f"[CenterGeoAttention] 调整 nhead: {nhead} -> {nhead_new} 以适配 d_model={d_model}")
            nhead = nhead_new

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.res_scale = float(res_scale)
        self.head_dropout = float(head_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.norm_c = nn.LayerNorm(d_model)
        self.norm_all = nn.LayerNorm(d_model)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.bias_dim = int(bias_dim)
        if self.bias_dim > 0:
            # 每个 head 一条 bias 线性
            self.bias_proj = nn.Linear(self.bias_dim, nhead, bias=False)
        else:
            self.bias_proj = None

        # 中心与所有节点特征融合的 gating-MLP
        self.fuse_mlp = mlp(
            din=d_model * 2,
            dh=d_model,
            dout=d_model,
            n=2,
            dropout=attn_dropout
        )
        self.gate = nn.Linear(d_model * 2, d_model)

    def _build_bias_features(self, h, rbf_ic=None, seqsep_ic=None,
                             nbr_idx=None, local_bias=None):
        """
        bias_feat: (N, D_bias_total)
        - rbf_ic:    (N, rbf_bins)
        - seqsep_ic: (N, seqsep_bins)
        - local_bias: (K, D_extra), K = len(nbr_idx)，仅在邻居位置上非零
        """
        pieces = []
        device, dtype = h.device, h.dtype
        N = h.size(0)

        if rbf_ic is not None:
            rbf_ic = rbf_ic.to(device=device, dtype=dtype)
            assert rbf_ic.size(0) == N
            pieces.append(rbf_ic)

        if seqsep_ic is not None:
            seqsep_ic = seqsep_ic.to(device=device, dtype=dtype)
            assert seqsep_ic.size(0) == N
            pieces.append(seqsep_ic)

        if (nbr_idx is not None) and (local_bias is not None) and (nbr_idx.numel() > 0):
            nbr_idx = nbr_idx.to(device=device, dtype=torch.long)
            local_bias = local_bias.to(device=device, dtype=dtype)
            K, d_extra = local_bias.shape
            assert nbr_idx.shape[0] == K, \
                f"local_bias.shape[0]={K}, nbr_idx.shape[0]={nbr_idx.shape[0]}"
            full_bias = h.new_zeros(N, d_extra)
            full_bias[nbr_idx] = local_bias
            pieces.append(full_bias)

        if not pieces:
            return None

        bias_feat = torch.cat(pieces, dim=-1)  # (N, D_bias_total)
        return bias_feat

    def forward(self, h, center_idx,
                rbf_ic=None, seqsep_ic=None,
                nbr_idx=None, local_bias=None):
        """
        input:
            h:         (N, d_model)
            center_idx:int
            rbf_ic:    (N, rbf_bins)
            seqsep_ic: (N, seqsep_bins)
            nbr_idx:   (K,)
            local_bias:(K, D_extra)
        output:
            h_new:     (N, d_model)   # 全部节点都会更新
        """
        assert h.dim() == 2, f"h shape={tuple(h.shape)}"
        N, d = h.size()
        assert d == self.d_model, f"d_model mismatch: {d} != {self.d_model}"

        c = int(center_idx)
        assert 0 <= c < N, f"center_idx={c}, N={N}"

        device, dtype = h.device, h.dtype

        # ====== 构造几何 / 序列 bias 特征 ======
        bias_feat = self._build_bias_features(
            h, rbf_ic=rbf_ic, seqsep_ic=seqsep_ic,
            nbr_idx=nbr_idx, local_bias=local_bias
        )  # (N, D_bias) or None

        # ====== 标准 multi-head attention on full graph ======
        h_c = self.norm_c(h[c:c + 1])   # (1, d_model)
        h_all = self.norm_all(h)        # (N, d_model)

        q = self.w_q(h_c)               # (1, d_model)
        k = self.w_k(h_all)             # (N, d_model)
        v = self.w_v(h_all)             # (N, d_model)

        H = self.nhead
        Hd = self.head_dim

        # reshape: (H, 1, Hd) / (H, N, Hd)
        q = q.view(1, H, Hd).transpose(0, 1)   # (H, 1, Hd)
        k = k.view(N, H, Hd).transpose(0, 1)   # (H, N, Hd)
        v = v.view(N, H, Hd).transpose(0, 1)   # (H, N, Hd)

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Hd)  # (H, 1, N)

        # ====== 加入 head-wise 几何 / 序列偏置 ======
        if (bias_feat is not None) and (self.bias_proj is not None):
            N_bias = bias_feat.size(0)
            D_in = int(self.bias_proj.in_features)

            if bias_feat.size(1) != D_in:
                # 理论上不会进这里，因为 GeoBlock 已经对齐了维度；这里只做兜底
                if bias_feat.size(1) < D_in:
                    pad = torch.zeros(N_bias, D_in - bias_feat.size(1),
                                      device=bias_feat.device, dtype=bias_feat.dtype)
                    bias_feat = torch.cat([bias_feat, pad], dim=-1)
                else:
                    bias_feat = bias_feat[:, :D_in]

            bias_head = self.bias_proj(bias_feat)              # (N, H)
            bias_head = bias_head.transpose(0, 1).unsqueeze(1) # (H, 1, N)
            attn_logits = attn_logits + bias_head

        # ====== softmax & dropout (全图) ======
        w = torch.softmax(attn_logits, dim=-1)             # (H, 1, N)
        w = self.attn_dropout(w)

        if self.training and self.head_dropout > 0.0:
            keep = (torch.rand(H, device=device) > self.head_dropout).to(dtype=w.dtype).view(H, 1, 1)
            w = w * keep
            denom = keep.sum().clamp_min(1.0) / H
            w = w / denom

        out_center = torch.matmul(w, v)                    # (H, 1, Hd)
        out_center = out_center.transpose(0, 1).reshape(1, self.d_model)  # (1, d_model)
        delta_c = self.w_o(out_center)                     # (1, d_model)

        h_c_new = h[c:c + 1] + self.res_scale * delta_c    # (1, d_model)

        # 对每个残基 i: concat[h_i, h_c_new] -> gate & fuse
        g = h_c_new.expand(N, self.d_model)                # (N, d_model)
        h_cat = torch.cat([h, g], dim=-1)                  # (N, 2*d_model)

        gate = torch.sigmoid(self.gate(h_cat))             # (N, d_model)
        fused = self.fuse_mlp(h_cat)                       # (N, d_model)
        delta_all = gate * fused                           # (N, d_model)

        h_new = h + self.res_scale * delta_all
        # 中心位点再覆盖一遍，保证数值稳定
        h_new[c:c + 1] = h_c_new

        return h_new



class GeoBlock(nn.Module):

    def __init__(self, d_node, d_edge, d_hidden, nhead_local = 3, dropout = 0.2,
                 update_coords = True, use_ffn = True, rbf_bins = 16, seqsep_bins = 4, orient_dims = 0,
                 rbf_centers = None, extra_local_bias = 0):
        super().__init__()
        self.egnn = EGNNLayer(d_node=d_node, d_edge=d_edge, d_hidden=d_hidden, 
                              update_coords=update_coords, dropout=dropout)

        self.rbf_bins = int(rbf_bins)
        self.seqsep_bins = int(seqsep_bins)
        self.orient_dims = int(orient_dims)

        # ✅ 关键：如果没手动指定，就让 extra_local_bias = rbf_bins + seqsep_bins，
        #    和 _build_aux_runtime 构造 local_bias 的维度对齐
        if extra_local_bias <= 0:
            extra_local_bias = self.rbf_bins + self.seqsep_bins
        self.extra_local_bias = int(extra_local_bias)

        bias_dim_total = self.rbf_bins + self.seqsep_bins + (self.orient_dims + self.extra_local_bias)

        self.center_attn = CenterGeoAttention(
            d_model=d_node,
            nhead=nhead_local,
            bias_dim=bias_dim_total,
            attn_dropout=dropout,
            head_dropout=0.0,
            res_scale=0.5
        )

        self.use_ffn = bool(use_ffn)
        self.ffn = mlp(d_node, 2 * d_node, d_node, dropout=dropout) if self.use_ffn else None


    @torch.no_grad()
    def _check_inputs(self, x, h, edge_index, edge_attr, center_idx, nbr_idx, local_bias, rbf_ic, seqsep_ic):
        N = h.size(0)
        assert x.shape == (N, 3), f"x shape={tuple(x.shape)}, expect (N,3)"
        assert edge_index.dim() == 2 and edge_attr.dim() == 2, \
            f"edge_index/edge_attr dims: {edge_index.dim()}, {edge_attr.dim()}"
        assert 0 <= int(center_idx) < N, f"center_idx={center_idx}, N={N}"

        if nbr_idx.numel() > 0:
            assert nbr_idx.dim() == 1 and nbr_idx.dtype == torch.long, \
                f"nbr_idx shape={tuple(nbr_idx.shape)}, dtype={nbr_idx.dtype}"
            assert local_bias.shape[0] == nbr_idx.shape[0], \
                f"local_bias.shape[0]={local_bias.shape[0]}, K={nbr_idx.shape[0]}"

        assert rbf_ic.shape[0] == N and seqsep_ic.shape[0] == N, \
            f"rbf_ic/seqsep_ic first dim must be N, got {rbf_ic.shape[0]}, {seqsep_ic.shape[0]}"

    def forward(self, x, h, edge_index, edge_attr, center_idx, nbr_idx, local_bias, rbf_ic, seqsep_ic):
        """
        input:  x:(N, 3)     h:(N, d_node)   edge_index:(2, E)   edge_attr:(E, d_edge)    center_idx: int
        nbr_idx: (K,)    local_bias: (K, D_extra)    rbf_ic: (N, rbf_bins)     seqsep_ic:(N, seqsep_bins)
        output:  x_new: (N, 3)  h_new: (N, d_node)
        """
        device, dtype = h.device, h.dtype

        x = x.to(device=device, dtype=dtype)
        h = h.to(device=device, dtype=dtype)
        edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_attr = edge_attr.to(device=device, dtype=dtype)
        nbr_idx = nbr_idx.to(device=device, dtype=torch.long)
        local_bias = local_bias.to(device=device, dtype=dtype)
        rbf_ic = rbf_ic.to(device=device, dtype=dtype)
        seqsep_ic = seqsep_ic.to(device=device, dtype=dtype)

        self._check_inputs(
            x, h, edge_index, edge_attr,
            center_idx, nbr_idx, local_bias, rbf_ic, seqsep_ic
        )

        x, h = self.egnn(x, h, edge_index, edge_attr)

        h = self.center_attn(
            h=h,
            center_idx=center_idx,
            rbf_ic=rbf_ic,
            seqsep_ic=seqsep_ic,
            nbr_idx=nbr_idx,
            local_bias=local_bias,
        )

        if self.ffn is not None:
            h = h + self.ffn(h)

        return x, h


class EGNNStab(nn.Module):
    """
    WT / MT 双塔（权重共享）:
      - 编码器: 若干 GeoBlock 串联(EGNN + CenterGeoAttention + FFN)
      - 读出:
          1) 分别得到 WT / MT 的节点特征 h_wt, h_mt
          2) 对每个节点做差 diff_i = h_mt[i] - h_wt[i]
          3) 在每个子图内，用可学习 attention pooling 对所有节点差分做加权求和
          4) 得到图级向量 z_graph → LayerNorm → Linear → ddG
    """

    def __init__(self, d_node, d_edge, depth=3, d_hidden=160, nhead_local=3, dropout=0.2,
                 update_coords=True, use_ffn=True, rbf_bins=16, seqsep_bins=4, orient_dims=0,
                 rbf_centers=None, extra_local_bias=0, hbond_channels=None, drop_hbond=False):

        super().__init__()
        self.d_node = int(d_node)
        self.d_edge = int(d_edge)
        self.depth = int(depth)
        self.rbf_bins = int(rbf_bins)
        self.seqsep_bins = int(seqsep_bins)
        self.orient_dims = int(orient_dims)
        self.extra_local_bias = int(extra_local_bias)

        # 输入节点特征投影到模型维度
        self.input_proj = nn.Linear(d_node, 48)
        self.d_model = 48

        # 哪些 edge 通道是 Hbond（可选消融）
        if hbond_channels is None:
            self.hbond_channels = None
        else:
            if isinstance(hbond_channels, (list, tuple)):
                self.hbond_channels = torch.tensor(hbond_channels, dtype=torch.long)
            elif isinstance(hbond_channels, torch.Tensor):
                self.hbond_channels = hbond_channels.to(dtype=torch.long)
            else:
                raise TypeError(f"hbond_channels 类型不支持: {type(hbond_channels)}")
        self.drop_hbond = bool(drop_hbond)

        # RBF centers（其实 encode 里又重新生成了，这里只是为兼容）
        if rbf_centers is None:
            rbf_centers = torch.linspace(0.0, 14.0, steps=rbf_bins)
        self.register_buffer("rbf_centers", rbf_centers.to(torch.float32), persistent=False)

        # GeoBlock 堆叠
        self.blocks = nn.ModuleList([
            GeoBlock(
                d_node=self.d_model,
                d_edge=d_edge,
                d_hidden=d_hidden,
                nhead_local=nhead_local,
                dropout=dropout,
                update_coords=update_coords,
                use_ffn=use_ffn,
                rbf_bins=rbf_bins,
                seqsep_bins=seqsep_bins,
                orient_dims=orient_dims,
                rbf_centers=self.rbf_centers,
                extra_local_bias=self.extra_local_bias,
            )
            for _ in range(depth)
        ])

        # 图级 diff 归一化 + 读出
        self.diff_norm = nn.LayerNorm(self.d_model)
        self.readout = nn.Linear(self.d_model, 1, bias=True)

        # learned attention pooling over node diffs
        self.pool_norm = nn.LayerNorm(self.d_model)
        self.pool_attn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 1)   # 每个节点一个 score
        )

    def reset_readout(self):
        w = getattr(self.readout, "weight", None)
        if isinstance(self.readout, LazyModuleMixin) or isinstance(w, UninitializedParameter):
            return
        nn.init.zeros_(self.readout.weight)
        if getattr(self.readout, "bias", None) is not None:
            nn.init.zeros_(self.readout.bias)

    @torch.no_grad()
    def _build_aux_runtime(self, graph, centers, local_radius: float = 8.0):
        device = graph["node_feats"].device
        dtype = graph["node_feats"].dtype

        x = graph["coords"].to(device=device, dtype=dtype)  # (N, 3)
        N = x.size(0)
        nums = graph.get("residue_nums", None)
        if nums is None:
            raise RuntimeError("[EGNNStab] graph 缺少 residue_nums, 无法构造 seqsep 偏置")
        nums = torch.as_tensor(nums, device=device, dtype=torch.long)

        centers = torch.as_tensor(centers, device=device, dtype=torch.long)  # (B,)
        B = int(centers.numel())
        rbf_bins = int(self.rbf_bins)
        seq_bins = int(self.seqsep_bins)

        rbf_centers = torch.linspace(0.0, 20.0, rbf_bins, device=device, dtype=dtype)
        gamma = 1.0 / ((rbf_centers[1] - rbf_centers[0] + 1e-8) ** 2)
        dists_all = torch.cdist(x[centers], x, p=2)  # (B, N)
        rbf_ic_all = torch.exp(-gamma * (dists_all[..., None] - rbf_centers[None, None, :]) ** 2)

        c_nums = nums[centers].view(B, 1)
        sep = (nums[None, :] - c_nums).abs()  # (B, N)
        s1 = (sep == 1)
        s2 = (sep >= 2) & (sep <= 4)
        s3 = (sep >= 5) & (sep <= 9)
        s4 = (sep >= 10)
        seqsep_ic_all = torch.stack([s1, s2, s3, s4], dim=-1).to(dtype=dtype)  # (B, N, 4)
        if seq_bins != 4:
            pad = torch.zeros(B, N, seq_bins, device=device, dtype=dtype)
            use = min(seq_bins, 4)
            pad[..., :use] = seqsep_ic_all[..., :use]
            seqsep_ic_all = pad

        nbr_lists, bias_rows = [], []
        for i in range(B):
            d = dists_all[i]  # (N,)
            nbr = torch.nonzero((d <= local_radius), as_tuple=False).flatten()
            nbr = nbr[nbr != centers[i]]
            if nbr.numel() == 0:
                idx = torch.topk(
                    d + (torch.arange(N, device=device) == centers[i]) * 1e6,
                    k=2,
                    largest=False,
                ).indices
                nbr = idx[idx != centers[i]][:1]
            bias_i = torch.cat(
                [rbf_ic_all[i][nbr], seqsep_ic_all[i][nbr]],
                dim=-1
            )  # (k_i, rbf_bins + seq_bins)
            nbr_lists.append(nbr)
            bias_rows.append(bias_i)

        center_ptr = torch.zeros(B + 1, dtype=torch.long, device=device)
        for i in range(B):
            center_ptr[i + 1] = center_ptr[i] + nbr_lists[i].numel()

        nbr_idx = torch.cat(nbr_lists, dim=0).to(torch.long)              # (K_total,)
        local_bias = torch.cat(bias_rows, dim=0).to(dtype=dtype)          # (K_total, rbf+seq)
        nbr_mask = torch.ones(nbr_idx.size(0), dtype=torch.bool, device=device)

        return {
            "center_ptr": center_ptr,
            "nbr_idx": nbr_idx,
            "local_bias": local_bias,
            "rbf_ic_all": rbf_ic_all,
            "seqsep_ic_all": seqsep_ic_all,
            "nbr_mask": nbr_mask,
        }

    def _maybe_drop_hbond(self, edge_attr):
        if not self.drop_hbond or self.hbond_channels is None:
            return edge_attr
        if self.hbond_channels.numel() == 0:
            return edge_attr
        ea = edge_attr.clone()
        idx = self.hbond_channels.to(device=ea.device)
        ea[:, idx] = 0.0
        return ea

    def encode(self, graph, aux):
        """
        只做编码，返回所有节点的最终特征:
            h_all: (N_total, d_model)
        pooling 放在 forward 里做（因为要用 WT/MT 差分）
        """
        x = graph["coords"]                    # (N, 3)
        h = self.input_proj(graph["node_feats"])   # (N, 48)
        ei = graph["edge_index"]
        ea = graph.get("edge_attr", None)
        assert ea is not None, "[encode] edge_attr 缺失，请检查数据层"

        device = h.device
        dtype = h.dtype

        ea = ea.to(device=device, dtype=dtype)
        ea = self._maybe_drop_hbond(ea)

        # 只走批路径（你的 collate 已经提供 centers + ptr）
        assert "centers" in graph and "ptr" in graph, \
            "[encode] 期望 batch 图包含 'centers' 和 'ptr'"

        centers = graph["centers"]
        ptr = graph["ptr"]

        def _get(k):
            return aux.get(k, graph.get(k, None))

        require = ["center_ptr", "nbr_idx", "local_bias", "rbf_ic_all", "seqsep_ic_all", "nbr_mask"]
        missing = [k for k in require if _get(k) is None]
        if missing:
            built = self._build_aux_runtime(graph, centers)
            for k, v in built.items():
                if k not in aux:
                    aux[k] = v

        center_ptr = aux["center_ptr"].to(device=device, dtype=torch.long)
        nbr_idx_all = aux["nbr_idx"].to(device=device, dtype=torch.long)
        local_bias_all = aux["local_bias"].to(device=device, dtype=dtype)
        rbf_all = aux["rbf_ic_all"].to(device=device, dtype=dtype)
        sep_all = aux["seqsep_ic_all"].to(device=device, dtype=dtype)

        B = int(centers.numel())

        # 每个样本的中心依次驱动 GeoBlock 更新同一组 x,h
        for i in range(B):
            c = int(centers[i].item())
            lo_nb, hi_nb = int(center_ptr[i].item()), int(center_ptr[i + 1].item())
            nbr_i = nbr_idx_all[lo_nb:hi_nb]
            bias_i = local_bias_all[lo_nb:hi_nb]
            rbf_i = rbf_all[i]      # (N, rbf_bins)
            sep_i = sep_all[i]      # (N, seqsep_bins)

            for blk in self.blocks:
                x, h = blk(
                    x, h, ei, ea,
                    center_idx=c,
                    nbr_idx=nbr_i,
                    local_bias=bias_i,
                    rbf_ic=rbf_i,
                    seqsep_ic=sep_i,
                )

        return h   # (N_total, d_model)

    def forward(self, wt, mt, aux_wt=None, aux_mt=None):
        """
        前向:
          1) 分别编码 WT / MT 得到所有节点特征 h_wt / h_mt
          2) 逐节点 diff = h_mt - h_wt
          3) 在每个子图内，对 diff 做可学习 attention pooling 得到图级向量
          4) 图级向量 → LayerNorm → Linear → ddG
        """
        if aux_wt is None:
            aux_wt = {}
        if aux_mt is None:
            aux_mt = {}

        h_wt = self.encode(wt, aux_wt)      # (N_total, d)
        h_mt = self.encode(mt, aux_mt)      # (N_total, d)

        diff_all = h_mt - h_wt              # (N_total, d_model)

        ptr = wt["ptr"]                     # (B+1,)
        B = ptr.numel() - 1

        zs = []
        for i in range(B):
            g_lo, g_hi = int(ptr[i].item()), int(ptr[i + 1].item())
            diff_i = diff_all[g_lo:g_hi]            # (Ni, d_model)

            # learned attention pooling over node diffs
            diff_i_norm = self.pool_norm(diff_i)    # (Ni, d)
            logits = self.pool_attn(diff_i_norm)    # (Ni, 1)
            alpha = torch.softmax(logits, dim=0)    # (Ni, 1)

            z_graph = (alpha * diff_i).sum(dim=0)   # (d_model,)
            z_graph = self.diff_norm(z_graph)       # 再做一遍 LayerNorm

            zs.append(z_graph)

        z = torch.stack(zs, dim=0)        # (B, d_model)
        y = self.readout(z).squeeze(-1)   # (B,)

        return y
