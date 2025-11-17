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


def _edge_uv(edge_index):
    """统一读取 edge_index: 返回 numpy 的 (u, v) 索引向量(int64)"""
    if isinstance(edge_index, torch.Tensor):
        U = edge_index[0].long().cpu().numpy()
        V = edge_index[1].long().cpu().numpy()
    else:
        U, V = edge_index
    return U.astype(np.int64), V.astype(np.int64)


def _unit_vec(v, eps = 1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


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


# ---------单位方向向量（N->CA, CA->C, CA->SC）---------
def feat_unit_dirs(n, ca, c, sc = None, mode = "frame", eps = 1e-6) :
    """
        input: 
            n, ca, c : (N, 3) 对应主链 N, CA, C 的坐标
            sc       : (N, 3) 侧链质心坐标
            mode="frame": 正交局部坐标系 [e1,e2,e3] 展平(9D)，稳健处理近共线退化
                        e1 = dir(CA->C)
                        e3 = normalize( e1 * dir(N->CA) )；若退化，改用参考轴 ref 与 e1 叉积
                        e2 = e3 * e1
                mode="dirs" : 三条方向向量展平(9D):[dir(N->CA), dir(CA->C), dir(CA->SC或0)]
        output:
            (N, 9) 的 float ndarray
    """


    if mode == "dirs":
        d1 = _unit_vec(ca - n)
        d2 = _unit_vec(c - ca)
        if sc is None:
            d3 = np.zeros_like(d1)
        else:
            d3 = _unit_vec(sc - ca)
        dirs = np.stack([d1, d2, d3], axis=-2)       # (N,3,3)
        return dirs.reshape(dirs.shape[0], 9)

    # mode == "frame"
    e1 = _unit_vec(c - ca)       # CA->C
    u2 = _unit_vec(ca - n)       # N->CA
    t  = np.cross(e1, u2)
    n_t = np.linalg.norm(t, axis=-1, keepdims=True)

    need_fallback = (n_t[..., 0] < eps)
    if np.any(need_fallback):
        ref = np.zeros_like(e1)
        x_axis = np.array([1.0, 0.0, 0.0], dtype=e1.dtype)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=e1.dtype)
        parallel_to_x = (np.abs((e1 * x_axis).sum(-1)) > 0.99).astype(bool)
        ref[~parallel_to_x] = x_axis
        ref[parallel_to_x]  = y_axis
        t_fb = np.cross(e1, ref)
        t[need_fallback] = t_fb[need_fallback]
        n_t = np.linalg.norm(t, axis=-1, keepdims=True)

    e3 = t / np.maximum(n_t, eps)
    e2 = np.cross(e3, e1)
    e2 = _unit_vec(e2, eps=eps)
    e3 = _unit_vec(e3, eps=eps)

    frame = np.stack([e1, e2, e3], axis=-2)  # (N,3,3)
    return frame.reshape(frame.shape[0], 9)


#  ---------序列距离 one-hot---------
def feat_edge_seq_distance_onehot( edge_index, residue_nums, max_sep = 20,
                                   include_overflow_bin = True):
    """
    input:
        edge_index : (2,E)
        residue_nums : List[int]
        max_sep   : int
        include_overflow_bin : True 时增加一个“>=max_sep”的溢出桶
    output:
        oh: (E, C) 
    """
    U, V = _edge_uv(edge_index)
    nums = np.asarray(residue_nums, dtype=int)
    sep = np.abs(nums[U] - nums[V]).astype(int)

    if include_overflow_bin:
        C = max_sep + 2
        idx = np.clip(sep, 0, max_sep + 1)       # max_sep+1 是溢出桶
        idx[sep <= max_sep] = sep[sep <= max_sep]
        idx[sep >  max_sep] = max_sep + 1
    else:
        C = max_sep + 1
        idx = np.clip(sep, 0, max_sep)

    oh = np.zeros((len(sep), C), dtype=float)
    if len(sep) > 0:
        oh[np.arange(len(sep)), idx] = 1.0
    return oh


#  ---------距离接触信号---------
def feat_edge_contact_binary( edge_index, ca, cutoff = 8.0):
    """
    输入:
        edge_index : (2,E)
        ca         : (N,3) CA坐标
        cutoff     : float, 接触阈值(Å)
    输出:
        contact: (E,1) 的 numpy 数组，∈ {0,1}
    """
    U, V = _edge_uv(edge_index)
    diff = ca[V] - ca[U]
    dist = np.linalg.norm(diff, axis=1)
    return (dist <= float(cutoff)).astype(float).reshape(-1, 1)


#  ---------局部坐标特征(15)---------
def feat_edge_local_frame_features( edge_index, ca, node_local_frames_9d):
    """
    input:
        edge_index            : (2,E)
        ca                    : (N,3) CA坐标
        node_local_frames_9d  : (N,9)
    output:
        feats: (E, 15) 的 numpy 数组，拼接顺序：
               [ dir_u(3), dir_v(3), rel_rot(9) ]
    """
    U, V = _edge_uv(edge_index)
    E = len(U)
    out = np.zeros((E, 15), dtype=float)

    R = node_local_frames_9d.reshape(-1, 3, 3)  # (N,3,3)

    r = ca[V] - ca[U]
    rnorm = np.linalg.norm(r, axis=1, keepdims=True)
    hat_uv = np.divide(r, rnorm, out=np.zeros_like(r), where=(rnorm > 1e-8))
    hat_vu = -hat_uv

    for e in range(E):
        Ru = R[U[e]]  # (3,3)
        Rv = R[V[e]]

        valid_u = np.linalg.norm(Ru, axis=(0,1)).sum() > 0.0
        valid_v = np.linalg.norm(Rv, axis=(0,1)).sum() > 0.0

        if valid_u:
            out[e, 0:3] = Ru.T @ hat_uv[e]  # (3,)
        if valid_v:
            out[e, 3:6] = Rv.T @ hat_vu[e]  # (3,)
        if valid_u and valid_v:
            Rel = Ru.T @ Rv                 # (3,3)
            out[e, 6:15] = Rel.reshape(-1)

    return out



#  ---------欧式距离---------
def feat_edge_distance(coords, edge_index):
    """
    input: coords (N,3)   edge_index (2,E)
    output: dist: (E,) 
    """
    if isinstance(edge_index, torch.Tensor):
        u = edge_index[0].long().cpu().numpy()
        v = edge_index[1].long().cpu().numpy()
    else:
        u, v = edge_index
    diff = coords[v] - coords[u]
    return np.linalg.norm(diff, axis=1)


# ---------距离的 RBF 展开---------
def feat_rbf(dist, num_centers = 16, cutoff = 20.0, gamma = None) :
    """
    input: dist (E,)  num_centers  cutoff   gamma
    output:  rbf: (E, num_centers) RBF 
    """
    centers = np.linspace(0.0, float(cutoff), int(num_centers), dtype=float)
    if gamma is None:
        delta = centers[1] - centers[0] if len(centers) > 1 else 1.0
        gamma = 1.0 / (delta**2 + 1e-12)
    D = dist[:, None] - centers[None, :]
    return np.exp(-gamma * (D ** 2))


# ---------夹角（平面夹角 ∠p1–p2–p3，返回 sin/cos）---------
def feat_angle_sincos(p1, p2, p3):
    """
    input: p1/p2/p3 (N,3)  ∠p1-p2-p3
    output: ang: (N,2) [sin(angle), cos(angle)]
    eg:
        - 链角: p1=CA[i-1], p2=CA[i], p3=CA[i+1]
        - 节点中心角: p1=neighborA, p2=center, p3=neighborB
    """
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v1u = np.divide(v1, n1, out=np.zeros_like(v1), where=(n1 > 1e-8))
    v2u = np.divide(v2, n2, out=np.zeros_like(v2), where=(n2 > 1e-8))
    dot = np.sum(v1u * v2u, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.arccos(dot)
    return np.stack([np.sin(ang), np.cos(ang)], axis=1)


# ---------邻居度数---------
def feat_degree(num_nodes, edge_index,  mode= "undirected",  normalize = False):
    """
    input: num_nodes: 节点数 N    edge_index: (2,E)     normalize: 归一化 [0,1]
            计算节点度。默认无向度（与 build_node_feature 中口径一致）。
            - mode="undirected": 统计 U 与 V 的并集计数(np.bincount(concat([U,V])))
            - mode="out"       : 仅统计 U(出度)
    output: deg: (N,)
    """
    if isinstance(edge_index, torch.Tensor):
        U = edge_index[0].long().cpu().numpy()
        V = edge_index[1].long().cpu().numpy()
    else:
        U, V = edge_index
    
    if mode == "undirected":
        idx = np.concatenate([U, V], axis=0)
    else:
        idx = U

    deg = np.bincount(idx, minlength=num_nodes).astype(np.float32)
    if normalize:
        maxv = float(np.max(deg)) if np.max(deg) > 0 else 1.0
        deg = deg / maxv
    return deg



# ---------接触类型 one-hot（边特征）---------
def feat_contact_onehot(edge_index, residue_nums, residue_ids, distances,  hbond_map,
                        short_cutoff= 5.0, long_cutoff = 8.0, seq_covalent_sep = 1) :
    """
    input:
        edge_index: (2,E)
        residue_nums: List[int] 
        residue_ids: List[str]
        distances: (E,) CA-CA 
        hbond_map: {(res_id_i, res_id_j): strength}
        short_cutoff/long_cutoff:
        seq_covalent_sep: |i-j|==1 
    output:
        contact_oh: (E, 4) one-hot
            [covalent, hbond, short_contact, long_contact]
    """
    if isinstance(edge_index, torch.Tensor):
        U = edge_index[0].long().cpu().numpy()
        V = edge_index[1].long().cpu().numpy()
    else:
        U, V = edge_index
    E = len(U)
    out = np.zeros((E, 4), dtype=float)

    resnums = np.asarray(residue_nums, dtype=int)
    seq_sep = np.abs(resnums[U] - resnums[V])
    cov_mask = (seq_sep == seq_covalent_sep)
    out[cov_mask, 0] = 1.0

    if hbond_map:
        rid_u = np.array(residue_ids, dtype=object)[U]
        rid_v = np.array(residue_ids, dtype=object)[V]
        for e in range(E):
            if (rid_u[e], rid_v[e]) in hbond_map:
                out[e, :] = 0.0
                out[e, 1] = 1.0

    non_cov_non_hb = (out[:, 0] == 0.0) & (out[:, 1] == 0.0)
    short_mask = non_cov_non_hb & (distances < short_cutoff)
    long_mask  = non_cov_non_hb & (distances >= short_cutoff) & (distances < long_cutoff)
    out[short_mask, 2] = 1.0
    out[long_mask, 3]  = 1.0

    return out


# ---------二面角/扭转角 (sin, cos)---------
def dihedral_sincos(p1, p2, p3, p4) :
    """
    input:
        p1/p2/p3/p4: (M,3)，四点定义的二面角，采用右手规则
    output:
        out: (M, 2) = [sin(dihedral), cos(dihedral)]
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # 法向
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # 归一化
    def _unit(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return np.divide(x, n, out=np.zeros_like(x), where=(n > 1e-8))

    n1u = _unit(n1)
    n2u = _unit(n2)
    b2u = _unit(b2)

    m1 = np.cross(n1u, b2u) 
    cosang = np.clip(np.sum(n1u * n2u, axis=1), -1.0, 1.0)
    sinang = np.sum(m1 * n2u, axis=1)
    return np.stack([sinang, cosang], axis=1)


# ---------主链 ω 二面角的正弦/余弦---------
def feat_backbone_omega_sincos(n, ca, c) :
    """
    input:
        n/ca/c: (N,3) 主链 N/CA/C 坐标
    output:
        omega_sc: (N, 2) = [sin(ω), cos(ω)]，其中
            ω_i = dihedral(C_i, N_{i+1}, CA_{i+1}, C_{i+1})
    """
    Nres = ca.shape[0]
    out = np.zeros((Nres, 2), dtype=float)
    if Nres < 2:
        return out
    p1 = c[:-1]
    p2 = n[1:]
    p3 = ca[1:]
    p4 = c[1:]
    sc = dihedral_sincos(p1, p2, p3, p4)
    out[1:] = sc
    return out


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


# ---------共价键标志（边级））---------
def feat_edge_covalent_flag( edge_index, residue_nums, seq_covalent_sep = 1) :
    """
    input:
        edge_index: (2,E) 
        residue_nums: List[int]
        seq_covalent_sep: |i-j| 
    output:
        cov_flag: (E,) ∈ {0,1}，该边是否为“共价边”
    """
    if isinstance(edge_index, torch.Tensor):
        U = edge_index[0].long().cpu().numpy()
        V = edge_index[1].long().cpu().numpy()
    else:
        U, V = edge_index
    resnums = np.asarray(residue_nums, dtype=int)
    seq_sep = np.abs(resnums[U] - resnums[V])
    return (seq_sep == int(seq_covalent_sep)).astype(float)


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


# ---------侧链质心距离差（边级）---------
def feat_edge_sidechain_distance_delta( edge_index, ca, sc, absolute = True) :
    """
    input:
        edge_index: (2,E)
        ca: (N,3) CA 坐标
        sc: (N,3) 侧链质心坐标
        absolute: True 返回 |d_SC - d_CA|; False 返回 (d_SC - d_CA)
    output:  delta: (E,) 侧链质心距离与 CA-CA 距离之差
    """
    if isinstance(edge_index, torch.Tensor):
        U = edge_index[0].long().cpu().numpy()
        V = edge_index[1].long().cpu().numpy()
    else:
        U, V = edge_index
    diff_ca = ca[V] - ca[U]
    d_ca = np.linalg.norm(diff_ca, axis=1)

    valid_u = np.isfinite(sc[U]).all(axis=1)
    valid_v = np.isfinite(sc[V]).all(axis=1)
    valid = valid_u & valid_v
    d_sc = np.zeros_like(d_ca)
    d_sc[valid] = np.linalg.norm(sc[V][valid] - sc[U][valid], axis=1)

    delta = d_sc - d_ca
    return np.abs(delta) if absolute else delta


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


def feat_edge_hbond_feature(edge_index, residue_ids, hbond_map, mode = "flag_strength",
                            min_strength = 0.0, undirected_match = True):
    """
    mode = ["none"|"flag"|"strength"|"flag_strength"]
      - "none"          -> (E,0)
      - "flag"          -> (E,1)
      - "strength"      -> (E,1)
      - "flag_strength" -> (E,2)
    """
    if isinstance(edge_index, torch.Tensor):
        E = int(edge_index.shape[1])
    else:
        U, V = edge_index
        E = len(U)

    if mode == "none" or not hbond_map:
        return np.zeros((E, 0), dtype=float)

    base = feat_edge_hbond_flag_strength(edge_index, residue_ids, hbond_map,
                                         min_strength=min_strength, undirected_match=undirected_match)
    if mode == "flag_strength":
        return base
    elif mode == "flag":
        return base[:, :1]
    elif mode == "strength":
        return base[:, 1:2]
    else:
        raise ValueError(f"Unknown mode={mode}")


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
    asa_map = None, rsa_map = None, ss_map = None, phi_map = None, psi_map = None,   
    include_node_hbond = False, hbond_map = None,  directed_pairs= None, 
    local_frame_mode = "frame",  local_density_radius=10.0, use_plddt=True ):

    """
    input:
        include_node_hbond : bool
        local_frame_mode : {"frame","dirs"}
            "frame"(右手正交坐标系); "dirs" 输出三条方向向量
    output:
        X_node : (N, D_node) 的 numpy.ndarray 或 torch.Tensor
        spec   : dict[str, tuple[int,int]] —— 每一段特征在 X_node 中的切片区间 [start, end)
                 具体维度如下（默认顺序）：
                 - res_onehot       : (N,20)  残基类型 one-hot
                 - physchem         : (N,6)   [charge, hydropathy, volume, sidechain_area, hb_prop, flexibility]
                 - asa_rsa          : (N,2)   [ASA, RSA]
                 - ss_onehot        : (N,8)   DSSP-8类 one-hot
                 - phi_psi_sincos   : (N,4)   [sinφ, cosφ, sinψ, cosψ]
                 - local_frame_9d   : (N,9)   正交局部坐标系 [e1,e2,e3] 展平
                 - degree           : (N,1)   
                 - local_density    : (N,1)   半径=10Å 的邻居计数
                 - omega_sincos     : (N,2)   主链 ω 的 [sinω, cosω]
                 - b_factor         : (N,1)
                 - hbond_counts     : (N,3)   [hb_total, hb_donor, hb_acceptor]
       维度(D_node)
        Base = 20 + 6 + 1 + 1 + 8 + 4 + 2 + 9 + 1 + 1 + 1 = 54
        - 不加节点氢键: D_node = 54
        - 加节点氢键(有 directed_pairs): D_node = 54 + 3 = 57
        - 加节点氢键(无 directed_pairs): D_node = 54 + 1 = 55
    """

    # ---------- 断言（输入一致性） ----------
    N = len(residue_ids)
    assert isinstance(sequence, str) and len(sequence) == N, "sequence 长度必须等于 N"
    for name, arr in [("ca", ca), ("n", n), ("c", c), ("sc_center", sc_center)]:
        assert isinstance(arr, np.ndarray) and arr.shape == (N, 3), f"{name} 必须是 (N,3) 的 numpy.ndarray"
        assert np.all(np.isfinite(arr)), f"{name} 含有 NaN/Inf"
    b = np.asarray(b_factors, dtype=float)
    assert b.shape == (N,), "b_factors 必须是 (N,)"; assert np.all(np.isfinite(b)), "b_factors 含有 NaN/Inf"
    # --- 将 b_factors 视作 pLDDT，并规范化到 [0,1] ---
    if use_plddt:
        if np.nanmax(b) > 1.5:  
            b = b / 100.0
    b = np.clip(b, 0.0, 1.0)

    if isinstance(edge_index, torch.Tensor):
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, "edge_index 必须为 (2,E)"
        U = edge_index[0].long().cpu().numpy(); V = edge_index[1].long().cpu().numpy()
    else:
        U, V = edge_index
        assert isinstance(U, (np.ndarray, list)) and isinstance(V, (np.ndarray, list)), "edge_index 应为 (U,V)"
        U = np.asarray(U); V = np.asarray(V)
        assert U.ndim == 1 and V.ndim == 1 and len(U) == len(V), "edge_index (U,V) 维度或长度不一致"
    assert U.size > 0 and V.size > 0, "edge_index 为空"
    assert U.min(initial=0) >= 0 and V.min(initial=0) >= 0 and U.max(initial=-1) < N and V.max(initial=-1) < N, \
        "edge_index 中存在越界下标"
    
    parts, spec = [], {}
    cursor = 0

    # 1) 氨基酸 one-hot (20)
    aa_onehot = feat_res_type_onehot(sequence).astype(float); assert aa_onehot.shape == (N, 20)
    parts.append(aa_onehot);  spec["aa_onehot20"] = (cursor, cursor+20); cursor += 20

    # 2) 理化性质 (6)
    physchem6 = feat_res_physchem(sequence).astype(float); assert physchem6.shape == (N, 6)
    parts.append(physchem6);  spec["physchem6"] = (cursor, cursor+6); cursor += 6

    # 3) ASA/RSA (1+1)
    if (asa_map is not None) and (rsa_map is not None):
        ASA, RSA = feat_asa_rsa(residue_ids, asa_map, rsa_map)
        ASA = np.asarray(ASA, dtype=float); RSA = np.asarray(RSA, dtype=float)
        assert ASA.shape == (N,) and RSA.shape == (N,), "ASA/RSA 形状错误"
    else:
        ASA = np.zeros((N,), dtype=float); RSA = np.zeros((N,), dtype=float)
    parts.append(ASA.reshape(-1,1));  spec["ASA"] = (cursor, cursor+1); cursor += 1
    parts.append(RSA.reshape(-1,1));  spec["RSA"] = (cursor, cursor+1); cursor += 1

    # 4) SS8 one-hot (8)
    ss_oh = feat_ss_onehot(residue_ids, ss_map or {}).astype(float); assert ss_oh.shape == (N, 8)
    parts.append(ss_oh);  spec["SS8_onehot"] = (cursor, cursor+8); cursor += 8

    # 5) φ/ψ 正余弦 (4)
    if (phi_map is not None) and (psi_map is not None):
        phi_psi = feat_phi_psi_sincos(residue_ids, phi_map, psi_map).astype(float)
        assert phi_psi.shape == (N, 4), "phi_psi_sincos 形状应为 (N,4)"
    else:
        phi_psi = np.zeros((N,4), dtype=float)
    parts.append(phi_psi);  spec["phi_psi(sc)"] = (cursor, cursor+4); cursor += 4

    # 6) ω 正余弦 (2)
    omega_sc = feat_backbone_omega_sincos(n, ca, c).astype(float); assert omega_sc.shape == (N, 2)
    parts.append(omega_sc);  spec["omega(sc)"] = (cursor, cursor+2); cursor += 2

    # 7) 局部坐标 9D
    local9 = feat_unit_dirs(n=n, ca=ca, c=c, sc=sc_center, mode=local_frame_mode).astype(float)
    assert local9.shape == (N, 9) and np.all(np.isfinite(local9)), "local_frame9 形状或数值异常"
    parts.append(local9);  spec["local_frame9"] = (cursor, cursor+9); cursor += 9

    # 8) degree（无向度)
    deg = np.bincount(np.concatenate([U, V]), minlength=N).astype(float).reshape(-1,1)
    parts.append(deg);  spec["degree"] = (cursor, cursor+1); cursor += 1

    # 9) 局部密度 (1)
    density = feat_local_density(ca, radius=float(local_density_radius)).astype(float).reshape(-1,1)
    assert density.shape == (N,1)
    parts.append(density);  spec["local_density"] = (cursor, cursor+1); cursor += 1

    # 10) B-factor (1)
    parts.append(b.reshape(-1,1))
    spec["plddt"] = (cursor, cursor+1)
    cursor += 1

    # 11) 节点氢键（唯一消融）
    if include_node_hbond:
        hb_total, hb_donor, hb_acceptor = feat_hbond_counts(residue_ids, hbond_map or {}, directed_pairs)
        hb_total = np.asarray(hb_total, dtype=float); hb_donor = np.asarray(hb_donor, dtype=float); hb_acceptor = np.asarray(hb_acceptor, dtype=float)
        assert hb_total.shape == (N,), "hb_total 形状错误"
        has_directed = (directed_pairs is not None) and (len(directed_pairs) > 0)
        if has_directed:
            hb = np.stack([hb_total, hb_donor, hb_acceptor], axis=1).astype(float)  # (N,3)
            parts.append(hb);  spec["hbond_node(total,donor,acceptor)"] = (cursor, cursor+3); cursor += 3
        else:
            parts.append(hb_total.reshape(-1,1));  spec["hbond_node(total)"] = (cursor, cursor+1); cursor += 1

    X_node = np.concatenate(parts, axis=1).astype(np.float32)
    assert X_node.shape[0] == N, "X_node 行数不等于 N"
    return X_node, spec


def build_edge_feature(edge_index, ca, residue_ids, residue_nums, n = None, c = None, sc = None,             
    node_local_frames_9d = None, include_edge_hbond= False, hbond_map = None,
    hbond_min_strength = 0.0, hbond_undirected = True,
    rbf_bins = 16, rbf_dmin = 2.0, rbf_dmax = 22.0, seq_sep_max= 20, local_frame_mode = "frame",
    include_edge_plddt=False, node_b_factors=None, plddt_mode="avg_abs"):

    """
    input:
        
    output:
        X_edge : (E, D_edge) 的 numpy.ndarray / torch.Tensor
        spec   : dict[name -> (start,end)]  
    维度(默认 rbf_num_centers=16、seq_onehot_max_sep=20):
        - dist_ca                 : (E,1)
        - rbf                     : (E,16)
        - seq_sep_onehot          : (E, 22)   # 0..20 + overflow(>=21)
        - contact_binary_<=8A     : (E,1)
        - local_dir_u             : (E,3)
        - local_dir_v             : (E,3)
        - local_rot_uv            : (E,9)
        - contact_onehot (可选)   : (E,4)   [covalent, hbond, short, long]
        - hbond_flag_strength     : (E,2)
        - sc_dist_delta           : (E,1)
        - covalent_flag           : (E,1)

       Base(无氢键)= 1(distance) + 16(RBF) + 22(seq one-hot) + 1(contact) + 1(covalent) + 15(local-frame) = 56
        - 不加边氢键: D_edge = 56
        - 加边氢键：  D_edge = 56 + 2 = 58
    """

    # ---------- 断言（输入一致性） ----------
    N = len(residue_ids)
    assert isinstance(ca, np.ndarray) and ca.shape == (N, 3) and np.all(np.isfinite(ca)), "ca 需为 (N,3) 且无 NaN/Inf"
    residue_nums = np.asarray(residue_nums)
    assert residue_nums.shape == (N,), "residue_nums 必须是 (N,)"

    if isinstance(edge_index, torch.Tensor):
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, "edge_index 必须为 (2,E)"
        U = edge_index[0].long().cpu().numpy(); V = edge_index[1].long().cpu().numpy()
    else:
        U, V = edge_index
        U = np.asarray(U); V = np.asarray(V)
        assert U.ndim == 1 and V.ndim == 1 and len(U) == len(V), "edge_index (U,V) 维度或长度不一致"
    E = len(U)
    assert E > 0, "edge_index 为空"
    assert U.min(initial=0) >= 0 and V.min(initial=0) >= 0 and U.max(initial=-1) < N and V.max(initial=-1) < N, \
        "edge_index 中存在越界下标"

    parts, spec = [], {}
    cursor = 0

    # 1) 距离 / RBF
    disp = ca[V] - ca[U]
    dist = np.linalg.norm(disp, axis=1).astype(np.float32); assert dist.shape == (E,) and np.all(np.isfinite(dist))
    parts.append(dist.reshape(-1, 1));  spec["distance"] = (cursor, cursor+1); cursor += 1

    centers = np.linspace(rbf_dmin, rbf_dmax, rbf_bins, dtype=np.float32)
    widths  = (centers[1] - centers[0]); gamma = 1.0 / (widths ** 2 + 1e-8)
    rbf = np.exp(-gamma * (dist[:, None] - centers[None, :]) ** 2).astype(np.float32)
    parts.append(rbf);  spec["rbf"] = (cursor, cursor+rbf.shape[1]); cursor += rbf.shape[1]

    # 2) 序列距离 one-hot
    seq_oh = feat_edge_seq_distance_onehot(edge_index, residue_nums,
                                           max_sep=seq_sep_max, include_overflow_bin=True)
    seq_oh = np.asarray(seq_oh, dtype=np.float32); assert seq_oh.shape[0] == E
    parts.append(seq_oh);  spec["seq_sep_onehot"] = (cursor, cursor+seq_oh.shape[1]); cursor += seq_oh.shape[1]

    # 3) 接触 flag（≤8Å）
    is_contact = (dist <= 8.0).astype(np.float32).reshape(-1, 1)
    parts.append(is_contact);  spec["contact_flag"] = (cursor, cursor+1); cursor += 1

    # 4) 共价 flag（|i-j|==1，单链）
    resnums = np.asarray(residue_nums, dtype=int)
    cov = (np.abs(resnums[U] - resnums[V]) == 1).astype(np.float32).reshape(-1, 1)
    parts.append(cov);  spec["covalent_flag"] = (cursor, cursor+1); cursor += 1

    # 6) 边 pLDDT（可选）
    if include_edge_plddt and (node_b_factors is not None):
        e_pl = feat_edge_plddt(edge_index, node_b_factors, mode=plddt_mode)  # (E,2) or (E,1)
        parts.append(e_pl)
        name = "edge_plddt(avg,absdiff)" if e_pl.shape[1] == 2 else "edge_plddt(avg)"
        spec[name] = (cursor, cursor + e_pl.shape[1]); cursor += e_pl.shape[1]


    # 7) 边氢键（唯一消融）
    if include_edge_hbond:
        hb = feat_edge_hbond_flag_strength(edge_index, residue_ids, hbond_map or {},
                                           min_strength=hbond_min_strength,
                                           undirected_match=hbond_undirected)  # (E,2)
        hb = np.asarray(hb, dtype=np.float32); assert hb.shape == (E, 2)
        parts.append(hb);  spec["hbond_edge(flag,strength)"] = (cursor, cursor+2); cursor += 2

    # 8) 局部坐标相对特征（15）= [3,3,9]
    if node_local_frames_9d is None:
        assert (n is not None) and (c is not None), "需要 node_local_frames_9d 或 (n,c) 以生成 9D 局部坐标"
        assert isinstance(n, np.ndarray) and isinstance(c, np.ndarray) and n.shape == c.shape == (N,3), "n/c 必须是 (N,3)"
        node_local_frames_9d = feat_unit_dirs(n=n, ca=ca, c=c, sc=sc, mode=local_frame_mode)
    node_local_frames_9d = np.asarray(node_local_frames_9d, dtype=float)
    assert node_local_frames_9d.shape == (N, 9) and np.all(np.isfinite(node_local_frames_9d)), "node_local_frames_9d 无效"

    local_feats = feat_edge_local_frame_features(edge_index, ca, node_local_frames_9d)  # (E,15)
    assert local_feats.shape == (E, 15) and np.all(np.isfinite(local_feats)), "local frame 边特征无效"
    parts.append(local_feats[:, 0:3]);  spec["local_dir_u"]  = (cursor, cursor+3); cursor += 3
    parts.append(local_feats[:, 3:6]);  spec["local_dir_v"]  = (cursor, cursor+3); cursor += 3
    parts.append(local_feats[:, 6:15]); spec["local_rot_uv"] = (cursor, cursor+9); cursor += 9

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





