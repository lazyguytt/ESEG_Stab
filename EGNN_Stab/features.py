import torch
import numpy as np
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

AA_LIST = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
]

AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}

AA_one_to_three = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}

AA_PROPERTIES = {
    'ALA': [0, 1.8, 88.6, 89.1, 0.25, 0.85],
    'CYS': [0, 2.5, 108.5, 121.2, 0.04, 0.70],
    'ASP': [-1, -3.5, 111.1, 133.1, 0.81, 0.75],
    'GLU': [-1, -3.5, 138.4, 147.1, 0.80, 0.78],
    'PHE': [0, 2.8, 189.9, 165.2, 0.02, 0.65],
    'GLY': [0, -0.4, 60.1, 75.1, 0.25, 0.95],
    'HIS': [1, -3.2, 153.2, 155.2, 0.50, 0.80],
    'ILE': [0, 4.5, 166.7, 131.2, 0.01, 0.45],
    'LYS': [1, -3.9, 168.6, 146.2, 0.76, 0.85],
    'LEU': [0, 3.8, 166.7, 131.2, 0.01, 0.50],
    'MET': [0, 1.9, 162.9, 149.2, 0.04, 0.70],
    'ASN': [0, -3.5, 114.1, 132.1, 0.65, 0.78],
    'PRO': [0, -1.6, 112.7, 115.1, 0.10, 0.25],
    'GLN': [0, -3.5, 143.8, 146.1, 0.70, 0.80],
    'ARG': [1, -4.5, 173.4, 174.2, 0.85, 0.88],
    'SER': [0, -0.8, 89.0, 105.1, 0.45, 0.85],
    'THR': [0, -0.7, 116.1, 119.1, 0.40, 0.80],
    'VAL': [0, 4.2, 140.0, 117.1, 0.02, 0.55],
    'TRP': [0, -0.9, 227.8, 204.2, 0.10, 0.75],
    'TYR': [0, -1.3, 193.6, 181.2, 0.15, 0.70],
}

SS_DICT = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}


# ---------突变信息分析---------
def get_mutation_info(mutation):
    """
        input: mutation 
        output: orig_res, mut_pos, mut_res (1,1,1)
    """
    try:
        if len(mutation) < 3:
            raise ValueError("Mutation string too short.")

        orig_res = mutation[0].upper()
        mut_res = mutation[-1].upper()
        mut_pos = int(mutation[1:-1])

        return orig_res, mut_pos, mut_res

    except Exception as e:
        logger.error(f"Invalid mutation format: '{mutation}' -> {e}")
        raise

# ---------残基类型 one-hot (20)---------
def feat_res_type_onehot(sequence):
    """
        input: sequence (N)
        output: onehot: (N, 20)
    """
    N = len(sequence)
    onehot = np.zeros((N, len(AA_LIST)), dtype=float)
    for i, aa1 in enumerate(sequence):
        aa3 = AA_one_to_three.get(aa1.upper(), None)
        if aa3 is None:
            continue
        idx = AA_TO_INDEX.get(aa3, None)
        if idx is not None:
            onehot[i, idx] = 1.0
    return onehot

# ---------残基理化性质 (6维)---------
def feat_res_physchem(sequence):
    """
        input: sequence (N)
        output: aa_properities (N, 6)  [charge, hydropathy, volume, sidechain_area, hb_prop, flexibility]
    """
    N = len(sequence)
    aa_properities = np.zeros((N, 6), dtype=float)
    for i, aa1 in enumerate(sequence):
        aa3 = AA_one_to_three.get(aa1.upper(), None)
        if aa3 and aa3 in AA_PROPERTIES:
            aa_properities[i] = np.asarray(AA_PROPERTIES[aa3], dtype=float)
    return aa_properities

# ---------溶剂可及性 ASA / RSA---------
def feat_asa_rsa(residue_ids, asa_map, rsa_map):
    """
        input: residue_ids, asa_map, rsa_map
        output: ASA (N), RSA(N)
    """
    N = len(residue_ids)
    ASA = np.array([float(asa_map.get(rid, 0.0)) for rid in residue_ids], dtype=float)
    RSA = np.array([float(rsa_map.get(rid, 0.0)) for rid in residue_ids], dtype=float)
    return ASA, RSA

# ---------二级结构 one-hot (DSSP-8类)---------
def feat_ss_onehot(residue_ids, ss_map):
    """
        input: residue_ids, ss_map
        output: ss_oh (N, C)   C = max(SS_DICT.values()) + 1
    """
    num_classes = int(max(SS_DICT.values())) + 1
    N = len(residue_ids)
    idx = np.full((N,), SS_DICT.get('-', num_classes - 1), dtype=int)

    for i, rid in enumerate(residue_ids):
        s = ss_map.get(rid, '-')
        idx[i] = SS_DICT.get(s, SS_DICT.get('-', num_classes - 1))

    ss_oh = np.zeros((N, num_classes), dtype=float)
    if N > 0:
        ss_oh[np.arange(N), idx] = 1.0
    return ss_oh

# ---------主链二面角 φ/ψ 的正余弦---------
def feat_phi_psi_sincos(residue_ids, phi_map, psi_map):
    """
        input: residue_ids, phi_map, psi_map
        output: ang (N, 4)   [sin(phi), cos(phi), sin(psi), cos(psi)]
    """
    phi = np.array([float(phi_map.get(rid, 0.0)) for rid in residue_ids], dtype=float)
    psi = np.array([float(psi_map.get(rid, 0.0)) for rid in residue_ids], dtype=float)
    rad_phi = np.deg2rad(phi)
    rad_psi = np.deg2rad(psi)
    ang = np.stack([np.sin(rad_phi), np.cos(rad_phi), np.sin(rad_psi), np.cos(rad_psi)], axis=1)
    return ang

#  ---------序列距离 one-hot---------
def feat_edge_seq_distance_onehot( edge_index, residue_nums, max_sep=20, include_overflow_bin=True):
    """
    output:
        oh: (E, C)  one-hot, C = max_sep+1(+1)
    """
    if isinstance(edge_index, torch.Tensor):
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, "edge_index 必须是 (2,E)"
        edge_np = edge_index.detach().cpu().numpy()
        U = edge_np[0]
        V = edge_np[1]
    else:
        if isinstance(edge_index, np.ndarray):
            assert edge_index.ndim == 2 and edge_index.shape[0] == 2, "edge_index 必须是 (2,E)"
            U, V = edge_index[0], edge_index[1]
        else:
            U, V = edge_index   # tuple/list
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)

    # ---- 计算序列间隔 ----
    nums = np.asarray(residue_nums, dtype=int)
    sep = np.abs(nums[U] - nums[V]).astype(int)
    E = len(sep)

    # ---- one-hot 编码 ----
    if include_overflow_bin:
        C = max_sep + 2          
        idx = np.clip(sep, 0, max_sep + 1)
        idx[sep <= max_sep] = sep[sep <= max_sep]
        idx[sep >  max_sep] = max_sep + 1
    else:
        C = max_sep + 1          # 0..max_sep
        idx = np.clip(sep, 0, max_sep)

    oh = np.zeros((E, C), dtype=float)
    if E > 0:
        oh[np.arange(E), idx] = 1.0
    return oh

# ---------氢键数（节点级：总数/供体数/受体数）---------
def feat_hbond_counts( residue_ids, hbond_map, directed_pairs = None) :
    """
    input:
        residue_ids: List[str]
        hbond_map: {(res_i, res_j): strength} ”
        directed_pairs: [(donor_id, acceptor_id, strength), ...] 
    output:
        hb_total:    (N,) 
        hb_donor:    (N,) 
        hb_acceptor: (N,) 
    """
    N = len(residue_ids)
    id2idx = {rid: i for i, rid in enumerate(residue_ids)}
    hb_total = np.zeros((N,), dtype=float)
    hb_donor = np.zeros((N,), dtype=float)
    hb_acceptor = np.zeros((N,), dtype=float)

    if hbond_map:
        neigh = {i: set() for i in range(N)}
        for (ri, rj) in hbond_map.keys():
            i = id2idx.get(ri, None); j = id2idx.get(rj, None)
            if i is None or j is None or i == j: 
                continue
            neigh[i].add(j); neigh[j].add(i)
        for i in range(N):
            hb_total[i] = float(len(neigh[i]))

    if directed_pairs:
        for (donor_id, acceptor_id, _s) in directed_pairs:
            i = id2idx.get(donor_id, None); j = id2idx.get(acceptor_id, None)
            if i is None or j is None or i == j:
                continue
            hb_donor[i] += 1.0
            hb_acceptor[j] += 1.0

    return hb_total, hb_donor, hb_acceptor

# ---------氢键标志 + 强度（边级）---------
def feat_edge_hbond_flag_strength( edge_index, residue_ids, hbond_map , min_strength = 0.0, undirected_match = True) :
    """
    input:
        edge_index: (2,E)
        residue_ids: List[str]
        hbond_map: {(res_i, res_j): strength(0..1)}
        min_strength: 低于该阈值的氢键视为不存在
        undirected_match: True 时，若 (u,v) 不在 map 中，会尝试匹配 (v,u)
    output:
        out: (E, 2)
             out[:,0] = hbond_flag ∈ {0,1}
             out[:,1] = hbond_strength ∈ [0,1]
    """
    if isinstance(edge_index, torch.Tensor):
        U = edge_index[0].long().cpu().numpy()
        V = edge_index[1].long().cpu().numpy()
    else:
        U, V = edge_index

    E = len(U)
    out = np.zeros((E, 2), dtype=float)
    if not hbond_map:
        return out

    rid_u = np.array(residue_ids, dtype=object)[U]
    rid_v = np.array(residue_ids, dtype=object)[V]
    for e in range(E):
        key = (rid_u[e], rid_v[e])
        s = hbond_map.get(key, None)
        if s is None and undirected_match:
            s = hbond_map.get((key[1], key[0]), None)
        if s is None or s < min_strength:
            continue
        out[e, 0] = 1.0
        out[e, 1] = float(s)
    return out

# ---------局部打包密度 / 邻居计数（节点级））---------
def feat_local_density(ca, radius = 8.0) :
    """
    output: (N,) 每个残基在半径 radius 内（不含自身）的邻居数
    """
    N = ca.shape[0]
    if N == 0: return np.zeros((0,), dtype=float)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ca)
        neigh = tree.query_ball_point(ca, r=float(radius))
        return np.array([len([j for j in lst if j != i]) for i, lst in enumerate(neigh)], dtype=float)
    except Exception:
        D = np.linalg.norm(ca[:,None,:] - ca[None,:,:], axis=-1)
        np.fill_diagonal(D, np.inf)
        return (D <= float(radius)).sum(axis=1).astype(float)


def feat_edge_plddt(edge_index, node_plddt, mode="avg_abs"):
    """
    根据结点 pLDDT 生成边特征:
      - mode="avg_abs": 返回 (E,2) = [ (plddt_u + plddt_v)/2 , |plddt_u - plddt_v| ]
      - mode="avg":     返回 (E,1) = [ (plddt_u + plddt_v)/2 ]
    """
    if isinstance(edge_index, torch.Tensor):
        U = edge_index[0].long().cpu().numpy()
        V = edge_index[1].long().cpu().numpy()
    else:
        U, V = edge_index
    p = np.asarray(node_plddt, dtype=float).reshape(-1)
    if np.nanmax(p) > 1.5:  # 兼容 0~100
        p = p / 100.0
    p = np.clip(p, 0.0, 1.0)
    pu, pv = p[U], p[V]
    if mode == "avg_abs":
        out = np.stack([0.5*(pu+pv), np.abs(pu-pv)], axis=1)
    elif mode == "avg":
        out = 0.5*(pu+pv)
        out = out.reshape(-1,1)
    else:
        raise ValueError(f"Unknown pLDDT mode: {mode}")
    return out.astype(np.float32)


def build_node_feature(sequence, residue_ids, ca, n, c, sc_center, edge_index, b_factors,
    asa_map=None, rsa_map=None, ss_map=None, phi_map=None, psi_map=None,
    include_node_hbond=False, hbond_map=None, directed_pairs=None, local_density_radius=10.0, use_plddt=True):
    """

      1) aa_onehot20        : (N,20)
      2) physchem6          : (N,6)
      3) ASA                : (N,1)
      4) RSA                : (N,1)
      5) SS8_onehot         : (N,8)
      6) phi_psi(sc)        : (N,4)
      7) degree             : (N,1)   # 基于无向度
      8) local_density      : (N,1)   # 半径=local_density_radius
      9) plddt              : (N,1)   # 由 b_factors 归一化而来

    => 基础维度 D_node_base = 43
    """

    # ---------- 基本检查 ----------
    N = len(residue_ids)
    assert isinstance(sequence, str) and len(sequence) == N, "sequence 长度必须等于 N"

    for name, arr in [("ca", ca), ("n", n), ("c", c), ("sc_center", sc_center)]:
        assert isinstance(arr, np.ndarray) and arr.shape == (N, 3), f"{name} 必须是 (N,3) 的 numpy.ndarray"
        assert np.all(np.isfinite(arr)), f"{name} 含有 NaN/Inf"

    b = np.asarray(b_factors, dtype=float)
    assert b.shape == (N,), "b_factors 必须是 (N,)"
    assert np.all(np.isfinite(b)), "b_factors 含有 NaN/Inf"

    # 处理 pLDDT / B-factor，统一到 [0,1]
    if use_plddt:
        if np.nanmax(b) > 1.5:  # 兼容 0~100
            b = b / 100.0
    b = np.clip(b, 0.0, 1.0)

    if isinstance(edge_index, torch.Tensor):
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, "edge_index 必须为 (2,E)"
        U = edge_index[0].long().cpu().numpy()
        V = edge_index[1].long().cpu().numpy()
    else:
        U, V = edge_index
        U = np.asarray(U)
        V = np.asarray(V)
        assert U.ndim == 1 and V.ndim == 1 and len(U) == len(V), "edge_index (U,V) 维度或长度不一致"

    assert U.size > 0 and V.size > 0, "edge_index 为空"
    assert U.min(initial=0) >= 0 and V.min(initial=0) >= 0 and U.max(initial=-1) < N and V.max(initial=-1) < N, \
        "edge_index 中存在越界下标"

    parts, spec = {}, {}
    feat_list = []
    cursor = 0

    # 1) 氨基酸 one-hot (20)
    aa_onehot = feat_res_type_onehot(sequence).astype(float)
    assert aa_onehot.shape == (N, 20)
    feat_list.append(aa_onehot)
    spec["aa_onehot20"] = (cursor, cursor + 20)
    cursor += 20

    # 2) 理化性质 (6)
    physchem6 = feat_res_physchem(sequence).astype(float)
    assert physchem6.shape == (N, 6)
    feat_list.append(physchem6)
    spec["physchem6"] = (cursor, cursor + 6)
    cursor += 6

    # 3) ASA (1) & 4) RSA (1)
    if (asa_map is not None) and (rsa_map is not None):
        ASA, RSA = feat_asa_rsa(residue_ids, asa_map, rsa_map)
        ASA = np.asarray(ASA, dtype=float)
        RSA = np.asarray(RSA, dtype=float)
        assert ASA.shape == (N,) and RSA.shape == (N,), "ASA/RSA 形状错误"
    else:
        ASA = np.zeros((N,), dtype=float)
        RSA = np.zeros((N,), dtype=float)

    feat_list.append(ASA.reshape(-1, 1))
    spec["ASA"] = (cursor, cursor + 1)
    cursor += 1

    feat_list.append(RSA.reshape(-1, 1))
    spec["RSA"] = (cursor, cursor + 1)
    cursor += 1

    # 5) SS8 one-hot (8)
    ss_oh = feat_ss_onehot(residue_ids, ss_map or {}).astype(float)
    assert ss_oh.shape == (N, 8)
    feat_list.append(ss_oh)
    spec["SS8_onehot"] = (cursor, cursor + 8)
    cursor += 8

    # 6) φ/ψ 正余弦 (4)
    if (phi_map is not None) and (psi_map is not None):
        phi_psi = feat_phi_psi_sincos(residue_ids, phi_map, psi_map).astype(float)
        assert phi_psi.shape == (N, 4), "phi_psi_sincos 形状应为 (N,4)"
    else:
        phi_psi = np.zeros((N, 4), dtype=float)

    feat_list.append(phi_psi)
    spec["phi_psi(sc)"] = (cursor, cursor + 4)
    cursor += 4

    # 7) degree（无向度）
    deg = np.bincount(np.concatenate([U, V]), minlength=N).astype(float).reshape(-1, 1)
    feat_list.append(deg)
    spec["degree"] = (cursor, cursor + 1)
    cursor += 1

    # 8) 局部密度 (1)
    density = feat_local_density(ca, radius=float(local_density_radius)).astype(float).reshape(-1, 1)
    assert density.shape == (N, 1)
    feat_list.append(density)
    spec["local_density"] = (cursor, cursor + 1)
    cursor += 1

    # 9) pLDDT (1)
    feat_list.append(b.reshape(-1, 1))
    spec["plddt"] = (cursor, cursor + 1)
    cursor += 1

    # 10) 节点氢键（可选 & 追加在末尾）
    if include_node_hbond:
        hb_total, hb_donor, hb_acceptor = feat_hbond_counts(
            residue_ids, hbond_map or {}, directed_pairs
        )
        hb_total = np.asarray(hb_total, dtype=float)
        hb_donor = np.asarray(hb_donor, dtype=float)
        hb_acceptor = np.asarray(hb_acceptor, dtype=float)
        assert hb_total.shape == (N,), "hb_total 形状错误"

        has_directed = (directed_pairs is not None) and (len(directed_pairs) > 0)
        if has_directed:
            hb = np.stack([hb_total, hb_donor, hb_acceptor], axis=1).astype(float)  # (N,3)
            feat_list.append(hb)
            spec["hbond_node(total,donor,acceptor)"] = (cursor, cursor + 3)
            cursor += 3
        else:
            feat_list.append(hb_total.reshape(-1, 1))
            spec["hbond_node(total)"] = (cursor, cursor + 1)
            cursor += 1

    X_node = np.concatenate(feat_list, axis=1).astype(np.float32)
    assert X_node.shape[0] == N, "X_node 行数不等于 N"
    return X_node, spec


def build_edge_feature(edge_index, ca, residue_ids, residue_nums,
    include_edge_hbond=False, hbond_map=None, hbond_min_strength=0.0, hbond_undirected=True,
    rbf_bins=16, rbf_dmin=2.0, rbf_dmax=22.0, seq_sep_max=20,
    include_edge_plddt=False, node_b_factors=None, plddt_mode="avg_abs"):
    """
      1) distance              : (E,1)   # CA-CA 距离
      2) rbf                   : (E,rbf_bins)    # 默认 16
      3) seq_sep_onehot        : (E, seq_sep_max+2)  # 0..max + overflow
      4) contact_flag          : (E,1)   # dist <= 8Å
      5) covalent_flag         : (E,1)   # |i-j| == 1 单链共价键
      6) edge_plddt            : (E,1 or 2)  # 可选，基于节点 pLDDT

    => 若 seq_sep_max=20, rbf_bins=16, plddt_mode="avg_abs":
         D_edge_base = 1 + 16 + 22 + 1 + 1 + 2 = 43
    """

    # ---------- 断言（输入一致性） ----------
    N = len(residue_ids)
    assert isinstance(ca, np.ndarray) and ca.shape == (N, 3) and np.all(np.isfinite(ca)), "ca 需为 (N,3) 且无 NaN/Inf"
    residue_nums = np.asarray(residue_nums)
    assert residue_nums.shape == (N,), "residue_nums 必须是 (N,)"

    if isinstance(edge_index, torch.Tensor):
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, "edge_index 必须为 (2,E)"
        U = edge_index[0].long().cpu().numpy()
        V = edge_index[1].long().cpu().numpy()
    else:
        U, V = edge_index
        U = np.asarray(U)
        V = np.asarray(V)
        assert U.ndim == 1 and V.ndim == 1 and len(U) == len(V), "edge_index (U,V) 维度或长度不一致"

    E = len(U)
    assert E > 0, "edge_index 为空"
    assert U.min(initial=0) >= 0 and V.min(initial=0) >= 0 and U.max(initial=-1) < N and V.max(initial=-1) < N, \
        "edge_index 中存在越界下标"

    parts = []
    spec = {}
    cursor = 0

    # 1) 距离 / RBF
    disp = ca[V] - ca[U]
    dist = np.linalg.norm(disp, axis=1).astype(np.float32)
    assert dist.shape == (E,) and np.all(np.isfinite(dist))
    parts.append(dist.reshape(-1, 1))
    spec["distance"] = (cursor, cursor + 1)
    cursor += 1

    centers = np.linspace(rbf_dmin, rbf_dmax, rbf_bins, dtype=np.float32)
    if len(centers) > 1:
        widths = centers[1] - centers[0]
    else:
        widths = 1.0
    gamma = 1.0 / (widths ** 2 + 1e-8)
    rbf = np.exp(-gamma * (dist[:, None] - centers[None, :]) ** 2).astype(np.float32)
    parts.append(rbf)
    spec["rbf"] = (cursor, cursor + rbf.shape[1])
    cursor += rbf.shape[1]

    # 2) 序列距离 one-hot
    seq_oh = feat_edge_seq_distance_onehot(
        edge_index,
        residue_nums,
        max_sep=seq_sep_max,
        include_overflow_bin=True,
    ).astype(np.float32)
    assert seq_oh.shape[0] == E
    parts.append(seq_oh)
    spec["seq_sep_onehot"] = (cursor, cursor + seq_oh.shape[1])
    cursor += seq_oh.shape[1]

    # 3) 接触 flag（≤8Å）
    is_contact = (dist <= 8.0).astype(np.float32).reshape(-1, 1)
    parts.append(is_contact)
    spec["contact_flag"] = (cursor, cursor + 1)
    cursor += 1

    # 4) 共价 flag（|i-j|==1，单链）
    resnums = np.asarray(residue_nums, dtype=int)
    cov = (np.abs(resnums[U] - resnums[V]) == 1).astype(np.float32).reshape(-1, 1)
    parts.append(cov)
    spec["covalent_flag"] = (cursor, cursor + 1)
    cursor += 1

    # 5) 边 pLDDT（可选）
    if include_edge_plddt and (node_b_factors is not None):
        e_pl = feat_edge_plddt(edge_index, node_b_factors, mode=plddt_mode).astype(np.float32)
        parts.append(e_pl)
        name = "edge_plddt(avg,absdiff)" if e_pl.shape[1] == 2 else "edge_plddt(avg)"
        spec[name] = (cursor, cursor + e_pl.shape[1])
        cursor += e_pl.shape[1]

    # 6) 边氢键（唯一消融，永远追加在末尾）
    if include_edge_hbond:
        hb = feat_edge_hbond_flag_strength(
            edge_index,
            residue_ids,
            hbond_map or {},
            min_strength=hbond_min_strength,
            undirected_match=hbond_undirected,
        ).astype(np.float32)  # (E,2)
        assert hb.shape == (E, 2)
        parts.append(hb)
        spec["hbond_edge(flag,strength)"] = (cursor, cursor + 2)
        cursor += 2

    X_edge = np.concatenate(parts, axis=1).astype(np.float32)
    assert X_edge.shape[0] == E and np.all(np.isfinite(X_edge)), "X_edge 存在 NaN/Inf 或行数不等于 E"
    return X_edge, spec


def build_edges(ca, radius = 18.0, bidirectional = True):
    """
    基于 CA 坐标的半径图构建 edge_index(2,E),不含自环。
    采用“有向表示的无向图”: bidirectional=True 时 (u,v) 与 (v,u) 同时存在。
    """
    N = int(ca.shape[0])
    if N == 0:
        return np.zeros((2, 0), dtype=np.int64)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ca)
        neigh = tree.query_ball_point(ca, r=float(radius))
    except Exception:
        D = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        neigh = [np.where((D[i] <= float(radius)) & (np.arange(N) != i))[0].tolist() for i in range(N)]

    edges = []
    for i, lst in enumerate(neigh):
        for j in lst:
            if i == j:
                continue
            edges.append((i, j))
            if bidirectional:
                edges.append((j, i))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    edges = np.unique(np.asarray(edges, dtype=np.int64), axis=0)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    return edges.T  # (2,E)


def debug_print_feature_layout():
    # ---- Node features ----
    print("\n===== NODE FEATURES LAYOUT (D_node = 43) =====")
    cursor = 0
    print(f"[0-19]   aa_onehot20")
    cursor = 20
    print(f"[20-25] physchem6")
    cursor = 26
    print(f"[26]    ASA")
    print(f"[27]    RSA")
    cursor = 28
    print(f"[28-35] SS8_onehot(8)")
    cursor = 36
    print(f"[36-39] phi_psi_sincos(4)")
    cursor = 40
    print(f"[40]    degree")
    print(f"[41]    local_density")
    print(f"[42]    plddt")

    print("\n===== Total Node Dim = 43 =====")

    # ---- Edge features ----
    print("\n===== EDGE FEATURES LAYOUT (D_edge = 45) =====")
    cursor = 0
    print(f"[0]     distance")
    print(f"[1-16]  RBF(16)")
    cursor = 17
    print(f"[17-38] seq_sep_onehot (22)")
    cursor = 39
    print(f"[39]    contact_flag")
    print(f"[40]    covalent_flag")
    print(f"[41-42] edge_plddt(avg, absdiff)")

    cursor = 43
    print("\n(Hbond edges not used → total = 45)")

    print("\n===== Total Edge Dim = 45 =====\n")



