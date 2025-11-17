import os
import math
import json
import torch
import shutil
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm.auto import tqdm as _tqdm
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple, Any, List
from Bio.PDB import PDBParser, is_aa, DSSP, Polypeptide
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from features import build_node_feature, build_edge_feature, build_edges, get_mutation_info
import warnings
import logging


# ========= 日志与常量 =========
warnings.filterwarnings("ignore", category=PDBConstructionWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_ASA = {
    'ALA': 115, 'ARG': 225, 'ASN': 160, 'ASP': 150,
    'CYS': 135, 'GLN': 180, 'GLU': 190, 'GLY': 75,
    'HIS': 195, 'ILE': 175, 'LEU': 170, 'LYS': 200,
    'MET': 185, 'PHE': 210, 'PRO': 145, 'SER': 115,
    'THR': 140, 'TRP': 255, 'TYR': 230, 'VAL': 155
}
MAX_ASA_DEFAULT = 160

one_to_three = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}
THREE_TO_ONE = Polypeptide.protein_letters_3to1


def has_any_nan(*arrays) :
    return any(not np.isfinite(a).all() for a in arrays)


def impute_missing_coords(ca, n, c, h,  sc, estimate_backbone_H = True):
    """
        只在发现 NaN 时调用：
        - N 缺失:  n = CA - 1.46Å * t
        - C 缺失:  c = CA + 1.53Å * t
        - H 缺失:  优先 v = (CA-N) + (N-C_prev), H = N + 1.0Å * v̂; 不可用则 H = (N或CA) + 1.0Å * t
        - SC 缺失: sc = CA
        其中 t 为基于 CA 的链切线单位向量
    """
    ca = np.asarray(ca, float)
    n  = np.asarray(n,  float).copy()
    c  = np.asarray(c,  float).copy()
    h  = np.asarray(h,  float).copy()
    sc = np.asarray(sc, float).copy()

    N = ca.shape[0]

    # ----------  计算链切线方向 t (N,3) ----------
    """
        用 CA 座标构造每个残基处的“链方向”单位向量 t[i]
        供后续对 N/C 缺失时做沿链方向的几何插补
    """
    t = np.zeros_like(ca)
    for i in range(N):
        if 0 < i < N - 1:
            d = ca[i + 1] - ca[i - 1]
        elif i == 0 and N > 1:
            d = ca[1] - ca[0]
        elif i == N - 1 and N > 1:
            d = ca[i] - ca[i - 1]
        else:
            d = np.array([1.0, 0.0, 0.0])
        norm = np.linalg.norm(d)
        t[i] = d / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])

    # ---------- 处理 N/C 坐标中的 NaN ----------
    missN = ~np.isfinite(n).all(axis=1)
    missC = ~np.isfinite(c).all(axis=1)
    if missN.any():
        n[missN] = ca[missN] - 1.46 * t[missN]
    if missC.any():
        c[missC] = ca[missC] + 1.53 * t[missC]

    # ---------- 处理 H 坐标中的 NaN ----------
    missH = ~np.isfinite(h).all(axis=1)
    if missH.any():
        idxs = np.where(missH)[0]
        for i in idxs:
            v = None
            if estimate_backbone_H and np.isfinite(n[i]).all():
                v = (ca[i] - n[i]).copy()
                if i > 0 and np.isfinite(c[i - 1]).all():
                    v += (n[i] - c[i - 1])
            if v is None or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-8:
                v = t[i]
            v = v / (np.linalg.norm(v) + 1e-8)
            base_n = n[i] if np.isfinite(n[i]).all() else ca[i]
            h[i] = base_n + 1.0 * v

    # ----------处理侧链中心 sc 中的 NaN ----------
    bad_sc = ~np.isfinite(sc).all(axis=1)
    if bad_sc.any():
        sc[bad_sc] = ca[bad_sc]

    return n, c, h, sc


def parse_pdb_for_features(pdb_path, include_sidechain_center = True,
                            estimate_backbone_H = True, drop_non_standard = True):
    """
    input:
        PDB_path (单链)
        include_sidechain_center:   是否计算侧链中心坐标
        estimate_backbone_H:        是否估算缺失的主链氢原子坐标
        drop_non_standard:          是否丢弃非标准氨基酸残基
    output: 
        ca_coords: (N,3)            Ca原子坐标,N为残基数量
        n_coords:  (N,3)            N原子坐标
        c_coords:  (N,3)            C原子坐标
        h_coords:  (N,3)            H原子坐标(可能包含估算值或NaN)
        sidechain_center: (N,3)     侧链重原子几何中心
        residue_ids: List[str]      e.g: "A_42_" or "A_42_A"
        residue_nums: List[int]     残基序号列表 
        b_factors: (N)              CA 的 B 因子(缺失置 0.0)
        seq1: str                   单字母序列(非标为 'X')
        residue_info: dict[str -> {res_num, chain_id, res_name, ins_code}]
    """

    structure = PDBParser(QUIET=True).get_structure("prot", pdb_path)
    if len(structure) == 0:
        raise ValueError("PDB parse error: structure is empty.")
    model = structure[0]
    chains = [ch for ch in model.get_chains()]
    if not chains:
        raise ValueError("PDB parse error: no chains found.")
    
    chains_with_ca = []
    for ch in chains:
        for res in ch:
            if "CA" in res:
                chains_with_ca.append(ch)
                break
    if len(chains_with_ca) != 1:
        raise ValueError(" Only supports a single chain PDB (exactly one chain containing CA). ")
    
    chain = chains_with_ca[0]
    chain_id = chain.id

    ca_coords, n_coords, c_coords, h_coords, side_coords = [], [], [], [], []
    residue_ids, residue_nums, b_factors, aa_1_list = [], [], [], []
    residue_info = {}

    for res in chain:
        if drop_non_standard:
            if not is_aa(res, standard=True):
                continue
        else:
            if "CA" not in res and not is_aa(res):
                continue
        if "CA" not in res:
            continue

        def get_coord(residue, atom_name):
            if atom_name not in residue:
                return None
            atom = residue[atom_name]
            if atom.is_disordered():
                try:
                    atom = atom.selected_child
                except Exception:
                    atom = list(atom.child_dict.values())[0]
            return atom.get_coord().astype(float)

        ca = get_coord(res, "CA")
        n  = get_coord(res, "N")
        c  = get_coord(res, "C") 

        if ca is None:
            continue 

        #---------- 残基标识与序列信息 ----------
        res_name = res.get_resname().upper()
        try:
            aa_1 = Polypeptide.three_to_one(res_name)
        except Exception:
            aa_1 = "X"
        res_num = int(res.id[1])
        ins_code = (res.id[2] or "").strip()
        res_id = f"{chain_id}_{res_num}_{ins_code}" if ins_code else f"{chain_id}_{res_num}_"

        # —---------- B 因子, 缺失置 0.0 ----------
        try:
            b = res["CA"].get_bfactor()
            b = float(b) if b is not None else 0.0
        except Exception:
            b = 0.0

        # ---------- 计算 h_coords (N,3) ----------
        h = None
        for hn in ("H", "HN", "H1", "H2", "H3"):
            h = get_coord(res, hn)
            if h is not None:
                break
    
        # ---------- 计算 sidechain_center (N,3) ----------
        if include_sidechain_center:
            sc = []
            for atom in res:
                an = atom.get_name().upper()
                if an in ("N", "CA", "C", "O"):
                    continue
                elem = (atom.element or "").upper() if hasattr(atom, "element") else ""
                if elem == "H" or an.startswith("H"):
                    continue
                try:
                    sc.append(atom.get_coord().astype(float))
                except Exception:
                    pass
            sc_center = np.mean(sc, axis=0) if len(sc) > 0 else np.array([np.nan, np.nan, np.nan])
        else:
            sc_center = ca

        ca_coords.append(ca)
        n_coords.append(n if n is not None else np.array([np.nan, np.nan, np.nan], float))
        c_coords.append(c if c is not None else np.array([np.nan, np.nan, np.nan], float))
        h_coords.append(h if h is not None else np.array([np.nan, np.nan, np.nan], float))
        side_coords.append(sc_center)

        residue_ids.append(res_id)
        residue_nums.append(res_num)
        b_factors.append(b)
        aa_1_list.append(aa_1)
        residue_info[res_id] = {
            "res_num": res_num,
            "chain_id": chain_id,
            "res_name": res_name,
            "ins_code": ins_code
        }

    if len(ca_coords) == 0:
        raise ValueError("No valid amino-acid residues with CA were found in this chain.")
    
    ca = np.array(ca_coords, dtype=float)
    n  = np.array(n_coords, dtype=float)
    c  = np.array(c_coords, dtype=float)
    h  = np.array(h_coords, dtype=float)
    sc = np.array(side_coords, dtype=float)

    if has_any_nan(n, c, h, sc):
        n, c, h, sc = impute_missing_coords(ca, n, c, h, sc, estimate_backbone_H=estimate_backbone_H)

    sequence = "".join(aa_1_list)

    return (
        ca, n, c, h, sc,                # (N,3)
        residue_ids,                                      # List[str], len=N
        residue_nums,                                     # List[int], len=N
        np.array(b_factors, dtype=float),                 # (N,)
        sequence,                                         # str
        residue_info                                      # dict[str -> {...}]
    )


def extract_dssp_features(pdb_path, target_chain = None):
    """
    input:PDB_path(单序列)
    output:
        ss_map:     DSSP二级结构字符
        asa_map:    绝对溶剂可及表面积 ASA (Å^2)
        rsa_map:    归一化溶剂可及表面积
        phi_map:    φ 角（缺失/非有限 -> 0.0)
        psi_map:    ψ 角（缺失/非有限 -> 0.0)
        hbond_map:  {(res_id_i, res_id_j) -> strength ∈ [0,1]} 
    """
    ss_map, asa_map, rsa_map, phi_map, psi_map = {}, {}, {}, {}, {}

    hbond_map = {}

    def _safe_float_nan(x):
        try:
            if x is None:
                return np.nan
            if isinstance(x, str) and x.strip().upper() in {"", "-", "NA", "NAN"}:
                return np.nan
            v = float(x)
            return v if math.isfinite(v) else np.nan
        except Exception:
            return np.nan


    def _fmt_res_id(chain_id, resnum, icode):
        icode_disp = (icode or '').strip()
        return f"{chain_id}_{resnum}_{icode_disp}" if icode_disp else f"{chain_id}_{resnum}_"

    def _energy_to_strength(E, E0 = 2.0):
        # DSSP: 能量 E(负值更强) → [0,1]
        try:
            e = float(E)
        except Exception:
            return 0.0
        return float(np.clip(max(0.0, -e) / E0, 0.0, 1.0))
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('temp', pdb_path)
        model = structure[0]

        chain_ids = [ch.id for ch in model.get_chains()]
        if target_chain is None:
            if len(chain_ids) > 1:
                raise ValueError(f"Detected multiple chains {chain_ids}. Please set target_chain explicitly.")
            target_chain = chain_ids[0]
        
        if shutil.which('mkdssp') is None:
            logger.warning("mkdssp not found; skip DSSP features.")
            return ss_map, asa_map, rsa_map, phi_map, psi_map, {}
        
        dssp = DSSP(model, pdb_path, dssp='mkdssp')

        dssp_entries = []  
        dssp_res_ids  = []  
        for key in dssp.keys():
            ch, (_, resnum, icode) = key
            if ch != target_chain:
                continue
            dssp_entries.append((key, dssp[key]))
            dssp_res_ids.append(_fmt_res_id(ch, resnum, icode))
            
        for (key, rec), res_id in zip(dssp_entries, dssp_res_ids):
            ch, (_, resnum, icode) = key

            # aa / ss / ASA / φ / ψ
            aa1 = rec[0]
            ss  = rec[1]
            asa = _safe_float_nan(rec[2])
            phi = _safe_float_nan(rec[3])
            psi = _safe_float_nan(rec[4])

            try:
                res = model[ch][(' ', resnum, icode if icode else ' ')]
                resname3 = res.resname.upper()
            except KeyError:
                aa1_up = (aa1 or 'X').upper()
                resname3 = one_to_three.get(aa1_up, 'UNK')

            max_asa = MAX_ASA.get(resname3, MAX_ASA_DEFAULT)
            rsa = asa / max_asa if max_asa > 0 else 0.0

            ss_char = ss if (ss is not None and ss != ' ') else '-'
            ss_map[res_id]  = ss_char
            asa_map[res_id] = asa
            rsa_map[res_id] = (asa / max_asa) if max_asa > 0 else np.nan
            phi_map[res_id] = phi
            psi_map[res_id] = psi

        energy_threshold = -0.5   
        min_seq_sep = 2           
        undirected = True         

        if dssp_entries and len(dssp_entries[0][1]) >= 13:
            def _try_add(p_src, rel_idx, energy):
                if rel_idx is None or energy is None:
                    return
                try:
                    rel = int(rel_idx)
                    e   = float(energy)
                except Exception:
                    return
                if e >= energy_threshold:
                    return
                p_dst = p_src + rel
                if p_dst < 0 or p_dst >= len(dssp_res_ids):
                    return
                i_id, j_id = dssp_res_ids[p_src], dssp_res_ids[p_dst]

                if abs(p_dst - p_src) < min_seq_sep:
                    return
                val = _energy_to_strength(e)

                hbond_map[(i_id, j_id)] = max(val, hbond_map.get((i_id, j_id), 0.0))
                if undirected:
                    hbond_map[(j_id, i_id)] = max(val, hbond_map.get((j_id, i_id), 0.0))

            for p, (_, rec) in enumerate(dssp_entries):
  
                _try_add(p, rec[5],  rec[6])   # NH→O_1
                _try_add(p, rec[7],  rec[8])   # O→NH_1
                _try_add(p, rec[9],  rec[10])  # NH→O_2
                _try_add(p, rec[11], rec[12])  # O→NH_2
        else:
            logger.warning("DSSP record lacks hbond fields; hbond_map left empty.")

        return ss_map, asa_map, rsa_map, phi_map, psi_map, hbond_map

    except Exception as e:
        logger.error(f"DSSP提取失败: {e}")
        return {}, {}, {}, {}, {}, {}


def load_ddg(excel_file) :
    df = pd.read_excel(excel_file)
    cols = set(df.columns)
    if "protein_id" in cols:
        protein = df["protein_id"].astype(str).str.strip()

    elif {"PDB_ID", "Chain"}.issubset(cols):
        pdb   = df["PDB_ID"].astype(str).str.strip().str.upper()
        chain = df["Chain"].astype(str).str.strip()
        protein = pdb + "_" + chain
    else:
        raise ValueError(
            f"{excel_file} 中既没有 'protein_id',"
            f"也没有 'PDB_ID'+'Chain' 这两列，当前列为: {list(df.columns)}"
        )
    if "Mutation" not in cols or "ddG" not in cols:
        raise ValueError(
            f"{excel_file} 中必须包含 'Mutation' 和 'ddG' 列，当前列为: {list(df.columns)}"
        )
    mut = df["Mutation"].astype(str).str.strip().str.upper()
    ddg = df["ddG"].astype(float)

    out = pd.DataFrame(
        {
            "protein_id": protein,
            "Mutation": mut,
            "ddG": ddg,
        }
    )
    print(f"📥 从 {os.path.basename(excel_file)} 读取 {len(out)} 条样本")
    return out


def crop_wt_mut_indices( ca_wt, residue_ids_wt,  residue_nums_wt, residue_ids_mt,       
    residue_nums_mt, mut_pos, base_radius= 12.0, min_nodes = 15) :
    """
    output:
      residue_ids_sub : List[str]        
      wt_seq_index    : np.ndarray(K,)  
      mt_seq_index    : np.ndarray(K,)   
      center_sub_idx  : int              裁剪后的 (0..K-1)
      radius_used     : float            最终裁剪半径
    """

    Nw = int(ca_wt.shape[0])
    assert Nw == len(residue_ids_wt) == len(residue_nums_wt), "WT 字段长度不一致"
    assert len(residue_ids_mt) == len(residue_nums_mt), "MT 字段长度不一致"
    assert Nw > 0, "WT 为空"
    K_target = int(max(1, min_nodes))          
    K_target = min(K_target, Nw)

    # --- 1) WT 中心定位 ---
    hits = np.where(np.asarray(residue_nums_wt, dtype=int) == int(mut_pos))[0]
    assert hits.size > 0, f"WT 中找不到突变位点 residue_num={mut_pos}"
    center_wt = int(hits[0])

    # --- 2) 计算半径（不足则一次性扩到第 K 近距离）---
    d = np.linalg.norm(ca_wt - ca_wt[center_wt], axis=1)
    order = np.lexsort((np.arange(Nw), d))
    if int((d <= float(base_radius)).sum()) >= K_target:
        radius_used = float(base_radius)
    else:
        kth = K_target - 1
        d_kth = float(d[order[kth]])
        radius_used = float(np.nextafter(d_kth, np.inf))

    #---- 3) 选 WT 子集（≤ radius_used，确保包含中心）----
    keep_idx_wt = order[d[order] <= radius_used]
    if center_wt not in keep_idx_wt:
        keep_idx_wt = np.insert(keep_idx_wt, 0, center_wt)
    keep_idx_wt = keep_idx_wt[np.lexsort((keep_idx_wt, d[keep_idx_wt]))].astype(np.int64)
    residue_ids_sub = [residue_ids_wt[i] for i in keep_idx_wt]

    # --- 4) MT 按 residue_id 一一映射到相同顺序---
    id2mt = {rid: i for i, rid in enumerate(residue_ids_mt)}
    try:
        keep_idx_mt = np.array([id2mt[rid] for rid in residue_ids_sub], dtype=np.int64)
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"MT 中缺少 WT 的残基 {missing}，请检查输入。")

    center_rid = residue_ids_wt[center_wt]
    center_sub_idx = residue_ids_sub.index(center_rid)

    return residue_ids_sub, keep_idx_wt, keep_idx_mt, int(center_sub_idx), float(radius_used)


def build_wt_mt_graph( wild_path, mutant_path, mutation_str, local_density_radius=8.0,include_edge_plddt=True, 
                      plddt_mode="avg_abs", include_edge_hbond=True, hbond_min_strength=0.0, hbond_undirected=True):
    """
    构建 WT/MT 两个蛋白质子图(先裁剪，再建图)
        "wt": {"node_feats":  (N, D_node) float32,
                "edge_index":  (2, E) int64,        # 双向有向、无自环
                "edge_attr": (E, D_edge) float32,
                "coords":      (N, 3) float32,      # CA 坐标（裁剪后）
                "residue_ids": List[str],
                "residue_nums":(N,) int64,
                "seq_index":   (K,) int64,          # 子图 → WT 原序列的索引映射
                "center_sub_idx": int
                "mutation "     :{sub_index, residue_id, wt_aa, mut_aa }}
        "mt": { ..... }
    """

    RADIUS = 14.0
    MIN_NODES = 15

    wt_aa, mut_pos, mt_aa = get_mutation_info(mutation_str)

    # ---------- 0) 解析 PDB ----------
    ca_wt, n_wt, c_wt, _, sc_wt, rid_wt, rnum_wt, b_wt, seq_wt, _ = parse_pdb_for_features(
        wild_path, include_sidechain_center=True, estimate_backbone_H=True, drop_non_standard=True
    )
    ca_mt, n_mt, c_mt, _, sc_mt, rid_mt, rnum_mt, b_mt, seq_mt, _ = parse_pdb_for_features(
        mutant_path, include_sidechain_center=True, estimate_backbone_H=True, drop_non_standard=True
    )

    # ---------- 1) DSSP ----------
    ss_wt, asa_wt, rsa_wt, phi_wt, psi_wt, hb_wt = extract_dssp_features(wild_path)
    ss_mt, asa_mt, rsa_mt, phi_mt, psi_mt, hb_mt = extract_dssp_features(mutant_path)

    # ---------- 2) 突变位点 ----------
    hits = np.where(np.asarray(rnum_wt, dtype=int) == int(mut_pos))[0]
    assert hits.size > 0, f"WT 中找不到突变位点 residue_num={mut_pos}"
    center_rid = rid_wt[int(hits[0])]

    # ---------- 3) WT 主导裁剪 + residue_id 对齐 ----------
    residue_ids_sub, wt_idx, mt_idx, _, radius_used = crop_wt_mut_indices(
        ca_wt=ca_wt,
        residue_ids_wt=rid_wt,
        residue_nums_wt=np.asarray(rnum_wt, dtype=int),
        residue_ids_mt=rid_mt,
        residue_nums_mt=np.asarray(rnum_mt, dtype=int),
        mut_pos=int(mut_pos),
        base_radius=RADIUS,
        min_nodes=MIN_NODES,
    )

    try:
        center_sub_idx = residue_ids_sub.index(center_rid)
    except ValueError:
        raise AssertionError("裁剪后的 residue_ids_sub 中未找到突变残基ID,数据不一致")

    # ---------- 4) 子图切片 ----------
    def _slice_side(ca, n, c, sc, b, rids, rnums, seq, keep):
        ca_s = ca[keep]
        n_s  = None if n  is None else n[keep]
        c_s  = None if c  is None else c[keep]
        sc_s = None if sc is None else sc[keep]
        b_s  = np.zeros((len(keep),), dtype=np.float32) if b is None else b[keep].astype(np.float32)
        rids_s  = [rids[i] for i in keep]
        rnums_s = np.asarray(rnums, dtype=int)[keep].astype(np.int64)
        seq_s   = "".join([seq[i] for i in keep])
        return ca_s, n_s, c_s, sc_s, b_s, rids_s, rnums_s, seq_s

    ca_w, n_w, c_w, sc_w, b_w, rids_w, rnums_w, seq_w_sub = _slice_side(
        ca_wt, n_wt, c_wt, sc_wt, b_wt, rid_wt, rnum_wt, seq_wt, wt_idx
    )
    ca_m, n_m, c_m, sc_m, b_m, rids_m, rnums_m, seq_m_sub = _slice_side(
        ca_mt, n_mt, c_mt, sc_mt, b_mt, rid_mt, rnum_mt, seq_mt, mt_idx
    )

    # ---------- 5) 构边 ----------
    edge_radius = max(RADIUS, radius_used)
    ei_w = build_edges(ca=ca_w, radius=edge_radius, bidirectional=True)
    ei_m = build_edges(ca=ca_m, radius=edge_radius, bidirectional=True)
    if isinstance(ei_w, tuple): ei_w = ei_w[0]
    if isinstance(ei_m, tuple): ei_m = ei_m[0]

    # ---------- 6) 节点/边特征 ----------
    Xn_w, _ = build_node_feature(
        sequence=seq_w_sub,
        residue_ids=rids_w,
        ca=ca_w,
        n=n_w,
        c=c_w,
        sc_center=sc_w,
        edge_index=ei_w,
        b_factors=b_w,
        asa_map=asa_wt,
        rsa_map=rsa_wt,
        ss_map=ss_wt,
        phi_map=phi_wt,
        psi_map=psi_wt,
        include_node_hbond=False,
        hbond_map=hb_wt,
        directed_pairs=None,
        local_density_radius=local_density_radius,
        use_plddt=True,
    )

    Xe_w, _ = build_edge_feature(
        edge_index=ei_w,
        ca=ca_w,
        n=n_w,             
        c=c_w,              
        residue_ids=rids_w,
        residue_nums=rnums_w,
        include_edge_hbond=include_edge_hbond,
        hbond_map=hb_wt,
        hbond_min_strength=hbond_min_strength,
        hbond_undirected=hbond_undirected,
        rbf_bins=16,
        rbf_dmin=2.0,
        rbf_dmax=22.0,
        seq_sep_max=20,
        include_edge_plddt=include_edge_plddt,
        node_b_factors=b_w,
        plddt_mode=plddt_mode,
    )

    Xn_m, _ = build_node_feature(
        sequence=seq_m_sub,
        residue_ids=rids_m,
        ca=ca_m,
        n=n_m,
        c=c_m,
        sc_center=sc_m,
        edge_index=ei_m,
        b_factors=b_m,
        asa_map=asa_mt,
        rsa_map=rsa_mt,
        ss_map=ss_mt,
        phi_map=phi_mt,
        psi_map=psi_mt,
        include_node_hbond=False,
        hbond_map=hb_mt,
        directed_pairs=None,
        local_density_radius=local_density_radius,
        use_plddt=True,
    )

    Xe_m, _ = build_edge_feature(
        edge_index=ei_m,
        ca=ca_m,
        n=n_m,             
        c=c_m,              
        residue_ids=rids_m,
        residue_nums=rnums_m,
        include_edge_hbond=include_edge_hbond,
        hbond_map=hb_mt,
        hbond_min_strength=hbond_min_strength,
        hbond_undirected=hbond_undirected,
        rbf_bins=16,
        rbf_dmin=2.0,
        rbf_dmax=22.0,
        seq_sep_max=20,
        include_edge_plddt=include_edge_plddt,
        node_b_factors=b_m,
        plddt_mode=plddt_mode,
    )


    mut_info = dict(
        sub_index=int(center_sub_idx),
        residue_id=str(center_rid),
        wt_aa=str(wt_aa),
        mt_aa=str(mt_aa),
    )
    wt_graph = dict(
        node_feats=Xn_w.astype(np.float32),
        edge_index=ei_w.astype(np.int64),
        edge_attr=Xe_w.astype(np.float32),
        coords=ca_w.astype(np.float32),
        residue_ids=rids_w,
        residue_nums=rnums_w.astype(np.int64),
        mutation=mut_info,
    )

    mt_graph = dict(
        node_feats=Xn_m.astype(np.float32),
        edge_index=ei_m.astype(np.int64),
        edge_attr=Xe_m.astype(np.float32),
        coords=ca_m.astype(np.float32),
        residue_ids=rids_m,
        residue_nums=rnums_m.astype(np.int64),
        mutation=dict(mut_info),
    )

    wt_graph["seq"] = seq_w_sub      
    mt_graph["seq"] = seq_m_sub 

    return {"wt": wt_graph, "mt": mt_graph}


class ProteinDataset(Dataset):
    """
        output: (__getitem__)
            obj : Dict[str, Any]
                {
                "wt": {"node_feats": (N,F), "edge_index": (2,E), "edge_attr": (E,D), ...},
                "mt": {"node_feats": (N,F), "edge_index": (2,E), "edge_attr": (E,D), ...},
                "ddg": float,
                "meta": {"signature": str, "wt_mtime": float, "mt_mtime": float}
                }
    """    
    def __init__(self, wild_dir, mutant_dir, disk_cache_dir, ddg_csv,
        normalize = False, norm_node_idx= None, norm_edge_idx = None,  
        cache_tag="v2_norm", rebuild=False,
        include_edge_plddt=True, plddt_mode="avg_abs",  include_edge_hbond=False,
        hbond_min_strength=0.0, hbond_undirected=True, local_density_radius=8.0,
        samples=None, stats_mode="auto"):

        super().__init__()

        self.cache_hits = 0; self.cache_miss = 0
        
        self.wild_dir = wild_dir
        self.mutant_dir = mutant_dir
        self.ddg_csv = ddg_csv
        self.cache_dir = disk_cache_dir
        self.graph_cache_dir = os.path.join(self.cache_dir, "graphData")
        
        self.normalize = bool(normalize)
        self.norm_node_idx = sorted(set(norm_node_idx or []))
        self.norm_edge_idx = sorted(set(norm_edge_idx or []))
        self.cache_tag = str(cache_tag)
        self.rebuild = bool(rebuild)

        self.include_edge_plddt = include_edge_plddt
        self.plddt_mode = plddt_mode
        self.include_edge_hbond = include_edge_hbond
        self.hbond_min_strength = float(hbond_min_strength)
        self.hbond_undirected = bool(hbond_undirected)
        self.local_density_radius = float(local_density_radius)

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.graph_cache_dir, exist_ok=True) 
        self.samples = self._load_pairs()
        if samples is not None:
            self.samples = list(samples)

        self.signature = json.dumps({
            "impl": "EGNN-protein.v1",
            "crop_radius": 14.0,
            "edge_radius": ">=crop",   
            "min_nodes": 15,
            "tag": self.cache_tag,
            "include_edge_plddt": bool(self.include_edge_plddt),
            "plddt_mode": str(self.plddt_mode),
            "include_edge_hbond": bool(self.include_edge_hbond),
            "hbond_min_strength": float(self.hbond_min_strength),
            "hbond_undirected": bool(self.hbond_undirected),
            "local_density_radius": float(self.local_density_radius),
        }, sort_keys=True)

        self.stats_mean_path = os.path.join(self.cache_dir, f"train_mean_{self.cache_tag}.pkl")
        self.stats_std_path  = os.path.join(self.cache_dir, f"train_std_{self.cache_tag}.pkl")

        self.node_mean = None
        self.node_std  = None
        self.edge_mean = None
        self.edge_std  = None

        if self.normalize:
            if stats_mode == "load-only":
                self._load_stats(strict=True)
            elif stats_mode == "compute":
                self._compute_and_save_stats_on_current_subset(use_tqdm=True)
            else:  # "auto"
                if not self._load_stats(strict=False):
                    self._compute_and_save_stats_on_current_subset(use_tqdm=True)


    def __len__(self):
        return len(self.samples)


    @torch.no_grad()
    def _compute_and_save_stats_on_current_subset(self, use_tqdm = False):

        if len(self.samples) == 0:
            raise RuntimeError("空数据集，无法计算统计。")

        normalize_backup = self.normalize
        self.normalize = False

        probe = self[0]
        d_node = int(probe["wt"]["node_feats"].shape[1])
        d_edge = int(probe["wt"]["edge_attr"].shape[1])

        node_mean = np.zeros(d_node, dtype=np.float64)
        node_m2   = np.zeros(d_node, dtype=np.float64)
        node_cnt  = np.zeros(d_node, dtype=np.int64)

        edge_mean = np.zeros(d_edge, dtype=np.float64)
        edge_m2   = np.zeros(d_edge, dtype=np.float64)
        edge_cnt  = np.zeros(d_edge, dtype=np.int64)

        idx_node = self.norm_node_idx
        idx_edge = self.norm_edge_idx

        def _acc(feat_tensor, idxs, mean, m2, cnt):
            if feat_tensor is None:
                return
            import numpy as _np
            try:
                import torch as _torch
                if _torch.is_tensor(feat_tensor):
                    x = feat_tensor.detach().cpu().numpy()
                elif isinstance(feat_tensor, _np.ndarray):
                    x = feat_tensor
                else:
                    x = _np.asarray(feat_tensor)
            except Exception:
                x = _np.asarray(feat_tensor)
            if x.size == 0:
                return
            if x.ndim == 1:  
                x = x.reshape(-1, 1)
            N, C = x.shape
            for j in idxs:
                if j >= C:
                    continue
                col = x[:, j].astype(_np.float64, copy=False)
                for v in col:
                    cnt[j] += 1
                    delta = v - mean[j]
                    mean[j] += delta / cnt[j]
                    delta2 = v - mean[j]
                    m2[j] += delta * delta2


        rng = range(len(self))
        if use_tqdm:
            try:
                from tqdm import tqdm as _tqdm
                rng = _tqdm(rng, desc="[Stats] computing mean/std on train split")
            except Exception:
                pass

        for i in rng:
            item = self[i] 
            for gkey in ("wt", "mt"):
                g = item[gkey]
                _acc(g["node_feats"], idx_node, node_mean, node_m2, node_cnt)
                _acc(g["edge_attr"], idx_edge, edge_mean, edge_m2, edge_cnt)

        node_var = np.zeros_like(node_mean)
        m = node_cnt > 1
        node_var[m] = node_m2[m] / (node_cnt[m] - 1)
        node_std = np.sqrt(np.maximum(node_var, 1e-16)).astype(np.float32)
        node_mean = node_mean.astype(np.float32)

        edge_var = np.zeros_like(edge_mean)
        m = edge_cnt > 1
        edge_var[m] = edge_m2[m] / (edge_cnt[m] - 1)
        edge_std = np.sqrt(np.maximum(edge_var, 1e-16)).astype(np.float32)
        edge_mean = edge_mean.astype(np.float32)

        node_std = np.clip(node_std, 1e-8, None)
        edge_std = np.clip(edge_std, 1e-8, None)

        self._save_stats_files(node_mean, node_std, edge_mean, edge_std)
        self.node_mean, self.node_std = node_mean, node_std
        self.edge_mean, self.edge_std = edge_mean, edge_std
        self.normalize = normalize_backup


    def __getitem__(self, i) :
        """
            读取/构图 → 统一命名 → 生成掩码并缺失置零 → （可选）标准化
        """

        pdb_id, mutation, ddg = self.samples[i]
        if "_" in pdb_id:
            pdb_base = pdb_id.split("_", 1)[0]   
        else:
            pdb_base = pdb_id 
    
        wild_path   = os.path.join(self.wild_dir,   f"{pdb_base}.pdb")
        mutant_path = os.path.join(self.mutant_dir, f"{pdb_base}_{mutation}.pdb")
        cache_path  = os.path.join(self.graph_cache_dir, f"{pdb_id}_{mutation}.pt")
        
        assert os.path.exists(wild_path),   f"缺失 wild pdb: {wild_path}"
        assert os.path.exists(mutant_path), f"缺失 mutant pdb: {mutant_path}"

        ok, reason = (False, "force_rebuild") if self.rebuild else self._cache_status(cache_path, wild_path, mutant_path)
        if ok:
            obj = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.cache_hits += 1
        else:
            self.cache_miss += 1
            graphs = build_wt_mt_graph(
                    wild_path, mutant_path, mutation,
                    local_density_radius=self.local_density_radius,
                    include_edge_plddt=self.include_edge_plddt,
                    plddt_mode=self.plddt_mode,
                    include_edge_hbond=self.include_edge_hbond,
                    hbond_min_strength=self.hbond_min_strength,
                    hbond_undirected=self.hbond_undirected
                )
            assert isinstance(graphs, dict) and "wt" in graphs and "mt" in graphs, "build_wt_mt_graph 返回结构异常"
            
            obj = {
                "wt": graphs["wt"],
                "mt": graphs["mt"],
                "ddg": float(ddg),
                "meta": {
                    "signature": self.signature,
                    "wt_mtime": os.path.getmtime(wild_path),
                    "mt_mtime": os.path.getmtime(mutant_path),
                }
            }
            torch.save(obj, cache_path)

        # === 统一命名 + 掩码处理 ===
        for side in ("wt", "mt"):
            g = obj[side]

            if "edge_attr" not in g:
                raise KeyError(f"{side}: graph missing 'edge_attr' (only 'edge_attr' is supported).")
            g["edge_attr"] = np.asarray(g["edge_attr"], dtype=np.float32)

            if "node_feat_mask" not in g:
                g["node_feat_mask"] = np.isfinite(g["node_feats"])
            if "edge_feat_mask" not in g:
                g["edge_feat_mask"] = np.isfinite(g["edge_attr"])

            g["node_feats"] = np.asarray(g["node_feats"], dtype=np.float32)
            g["edge_attr"]  = np.asarray(g["edge_attr"],  dtype=np.float32)

            g["node_feats"][~g["node_feat_mask"]] = 0.0
            g["edge_attr"][~g["edge_feat_mask"]] = 0.0

            g["node_ch_mask"] = np.any(g["node_feat_mask"], axis=0).astype(np.float32)
            g["edge_ch_mask"] = np.any(g["edge_feat_mask"], axis=0).astype(np.float32)

            assert np.isfinite(g["node_feats"]).all(), f"{side}: node_feats 仍含非有限值（掩码后）"
            assert np.isfinite(g["edge_attr"]).all(),  f"{side}: edge_attr 仍含非有限值（掩码后）"

        assert np.isfinite(obj["ddg"]), f"ddg 非有限值: {obj['ddg']}"
        if self.normalize:
            self._apply_normalization_inplace(obj["wt"])
            self._apply_normalization_inplace(obj["mt"])
        
        obj["seq"] = obj["wt"].get("seq", None)

        return obj
    

    @staticmethod
    def _concat_graphs(graphs, centers_local=None):
        """
            将同一侧(WT 或 MT)的多个小图拼接为一个大图，返回 PyTorch 张量：
            - node_feats: (N_all, F)
            - edge_attr:  (E_all, D)
            - edge_index: (2, E_all)
            - coords:     (N_all, 3)
            - batch:      (N_all,)   每个节点对应的图 id
            - node_ch_mask / edge_ch_mask: 通道级掩码(取所有图的与,只有当该通道在所有图都非缺失时为1)
            另外返回辅助切片以便可选的图内 pooling(node_slices/edge_slices)
        """
        assert len(graphs) >= 1, "graphs 不能为空"
        Ns = [int(g["node_feats"].shape[0]) for g in graphs]
        Es = [int(g["edge_attr"].shape[0])  for g in graphs]
        d_node = int(graphs[0]["node_feats"].shape[1])
        d_edge = int(graphs[0]["edge_attr"].shape[1])

        for g in graphs:
            assert g["node_feats"].shape[1] == d_node, f"node_feats 列数不一致: got {g['node_feats'].shape[1]} vs {d_node}"
            assert g["edge_attr"].shape[1]  == d_edge, f"edge_attr 列数不一致: got {g['edge_attr'].shape[1]} vs {d_edge}"

        def as_tensor(x, dtype=None):
            t = torch.as_tensor(x)
            if dtype is not None:
                t = t.to(dtype=dtype)
            return t

        ptr = torch.zeros(len(Ns) + 1, dtype=torch.long)
        ptr[1:] = torch.tensor(Ns, dtype=torch.long).cumsum(0)
        batch_vec = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(Ns)], dim=0)

         # ---- 节点/边特征与掩码 ----
        node_feats = torch.cat([as_tensor(g["node_feats"], dtype=torch.float32) for g in graphs], dim=0)
        node_mask = (torch.cat([as_tensor(g["node_feat_mask"], dtype=torch.float32) for g in graphs], dim=0)
                    if "node_feat_mask" in graphs[0] and graphs[0]["node_feat_mask"] is not None
                    else torch.ones((ptr[-1].item(), d_node), dtype=torch.float32))

        edge_attr = torch.cat([as_tensor(g["edge_attr"], dtype=torch.float32) for g in graphs], dim=0)
        edge_mask = (torch.cat([as_tensor(g["edge_feat_mask"], dtype=torch.float32) for g in graphs], dim=0)
                    if "edge_feat_mask" in graphs[0] and graphs[0]["edge_feat_mask"] is not None
                    else torch.ones((sum(Es), d_edge), dtype=torch.float32))

        coords = torch.cat([as_tensor(g["coords"], dtype=torch.float32) for g in graphs], dim=0)

        # ---- edge_index 拼接（带节点偏移）----
        def to_ei(ei):
            ei = as_tensor(ei, dtype=torch.long)
            if ei.dim() == 2:
                if ei.shape[0] == 2:  return ei
                if ei.shape[1] == 2:  return ei.t().contiguous()
            raise ValueError("edge_index 必须是形状 (2,E) 或 (E,2)")
        eis, offset = [], 0
        for i, g in enumerate(graphs):
            ei = to_ei(g["edge_index"])
            assert ei.numel() > 0 and ei.shape[1] > 0, "子图出现空边，请检查裁剪半径/坐标有效性"
            ei = ei + offset
            eis.append(ei)
            offset += Ns[i]
        edge_index = torch.cat(eis, dim=1)

        # ---- residue_nums（可选）----
        residue_nums = None
        if "residue_nums" in graphs[0] and graphs[0]["residue_nums"] is not None:
            residue_nums = torch.cat([as_tensor(g["residue_nums"], dtype=torch.long) for g in graphs], dim=0)

        # ---- 通道级掩码（AND 语义：仅当所有子图该列均有有效值时为 1）----
        node_ch_mask = torch.stack([
            torch.as_tensor(g.get("node_feat_mask", np.ones_like(g["node_feats"], dtype=bool))).any(dim=0)
            for g in graphs
        ], dim=0).all(dim=0).to(torch.float32)

        edge_ch_mask = torch.stack([
            torch.as_tensor(g.get("edge_feat_mask", np.ones_like(g["edge_attr"], dtype=bool))).any(dim=0)
            for g in graphs
        ], dim=0).all(dim=0).to(torch.float32)

        out = dict(
            node_feats=node_feats,
            edge_attr=edge_attr,
            edge_index=edge_index,
            coords=coords,
            batch=batch_vec,
            ptr=ptr,
            node_feat_mask=node_mask,
            edge_feat_mask=edge_mask,
            node_ch_mask=node_ch_mask,
            edge_ch_mask=edge_ch_mask,
        )

        if residue_nums is not None:
            out["residue_nums"] = residue_nums

        if centers_local is not None:
            local_c = torch.as_tensor(centers_local, dtype=torch.long)
            assert local_c.numel() == len(graphs), "centers_local 长度必须与子图数量一致"
            out["centers"] = ptr[:-1] + local_c

        return out


    @staticmethod
    def collate_fn(batch) :
        """
        将若干样本打包成一个 batch(图拼接版):
        返回：
        {
          "wt": 大拼图(dict of torch.Tensor),
          "mt": 大拼图(dict of torch.Tensor),
          "ddg": (B,) torch.float32
        }
        """
        wt_centers_local = [int(b["wt"]["mutation"]["sub_index"]) for b in batch]
        mt_centers_local = [int(b["mt"]["mutation"]["sub_index"]) for b in batch]

        wt_big = ProteinDataset._concat_graphs([b["wt"] for b in batch],
                                            centers_local=wt_centers_local)
        mt_big = ProteinDataset._concat_graphs([b["mt"] for b in batch],
                                            centers_local=mt_centers_local)

        ddg = torch.tensor([b["ddg"] for b in batch], dtype=torch.float32)

        seqs = [b.get("seq", None) for b in batch]
        return {"wt": wt_big, "mt": mt_big, "ddg": ddg, "seq": seqs}


    def _load_pairs(self):
        table = load_ddg(self.ddg_csv)
        out = []
        try:
            import pandas as pd
        except ImportError:
            pd = None

        if pd is not None and isinstance(table, pd.DataFrame):
            colmap = {c.lower(): c for c in table.columns}
            id_col = colmap.get("pdb_id") or colmap.get("protein_id")
            if id_col is None:
                raise ValueError(
                    f"ddg 表中找不到 'pdb_id' 或 'protein_id' 列，当前列为: {list(table.columns)}"
                )

            mut_col = colmap.get("mutation") or colmap.get("mut")
            if mut_col is None:
                raise ValueError(
                    f"ddg 表中找不到 'Mutation' / 'mutation' / 'mut' 列，当前列为: {list(table.columns)}"
                )

            ddg_col = None
            for k in ["ddg", "ddG".lower()]: 
                if k in colmap:
                    ddg_col = colmap[k]
                    break
            if ddg_col is None:
                for c in table.columns:
                    if c.lower() == "ddg":
                        ddg_col = c
                        break
            if ddg_col is None:
                raise ValueError(
                    f"ddg 表中找不到 ddG 列 (例如 'ddG' / 'ddg')，当前列为: {list(table.columns)}"
                )
            for _, r in table.iterrows():
                pid = str(r[id_col])
                mut = str(r[mut_col])
                ddg = float(r[ddg_col])
                out.append((pid, mut, ddg))

            return out

        if isinstance(table, (list, tuple)) and len(table) > 0:
            first = table[0]
            if isinstance(first, (list, tuple)) and len(first) == 3:
                return [(str(a), str(b), float(c)) for (a, b, c) in table]

        raise TypeError(
            f"_load_pairs: 不识别的 ddg 数据格式 type(table)={type(table)}; "
            f"请检查 load_ddg 返回类型。"
        )


    def _cache_status(self, cache_path, wt_path, mt_path):
        """检查缓存并给出 miss 原因。"""
        if not os.path.exists(cache_path):
            return False, "missing"
        try:
            obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            return False, "load_error"
        meta = obj.get("meta", {})
        if meta.get("signature") != self.signature:
            return False, "signature_mismatch"
        wt_ok = abs(meta.get("wt_mtime", -1) - os.path.getmtime(wt_path)) < 1e-6
        mt_ok = abs(meta.get("mt_mtime", -1) - os.path.getmtime(mt_path)) < 1e-6
        if not (wt_ok and mt_ok):
            return False, "mtime_mismatch"
        return True, "ok"


    def _apply_normalization_inplace(self, obj):

        def _get_channel_mask(obj, key_mask, dim):
            mask = obj.get(key_mask, None)
            if mask is None:
                alt = obj.get("node_feat_mask" if "node" in key_mask else "edge_feat_mask", None)
                if alt is not None:
                    if torch.is_tensor(alt):
                        alt = alt.detach().cpu().numpy()
                    alt = np.asarray(alt, dtype=bool)
                    if alt.ndim == 2:
                        col = alt.any(axis=0)
                    else:
                        col = alt.reshape(-1)
                else:
                    col = np.ones(dim, dtype=bool)
            else:
                if torch.is_tensor(mask):
                    mask = mask.detach().cpu().numpy()
                col = np.asarray(mask, dtype=bool).reshape(-1)

            if col.size < dim:
                pad = np.ones(dim - col.size, dtype=bool)
                col = np.concatenate([col, pad], axis=0)
            elif col.size > dim:
                col = col[:dim]
            return col

        # ================= Node =================
        h = obj.get("node_feats", None)
        if h is not None and self.node_mean is not None and self.node_std is not None:
            Dn = h.shape[1]
            idx_all = self._as_list(self.norm_node_idx) or []
            idx_all = np.asarray(idx_all, dtype=int)

            mask_full = _get_channel_mask(obj, "node_ch_mask", Dn)

            valid_idx = idx_all[(idx_all < Dn)]
            sel = mask_full[valid_idx]          
            cols = valid_idx[sel].astype(int) 

            if cols.size > 0:
                node_mean = np.asarray(self.node_mean, dtype=np.float32)
                node_std  = np.clip(np.asarray(self.node_std, dtype=np.float32), 1e-8, None)
                if torch.is_tensor(h):
                    mean_t = torch.as_tensor(node_mean[cols], dtype=h.dtype, device=h.device)
                    std_t  = torch.as_tensor(node_std[cols],  dtype=h.dtype, device=h.device)
                    h[:, cols] = (h[:, cols] - mean_t) / std_t
                else:
                    h[:, cols] = (h[:, cols] - node_mean[cols]) / node_std[cols]

        # ================= Edge =================
        e = obj.get("edge_attr", None)
        if e is not None and self.edge_mean is not None and self.edge_std is not None:
            De = e.shape[1]
            idx_all = self._as_list(self.norm_edge_idx) or []
            idx_all = np.asarray(idx_all, dtype=int)

            mask_full = _get_channel_mask(obj, "edge_ch_mask", De)

            valid_idx = idx_all[(idx_all < De)]
            sel = mask_full[valid_idx]
            cols = valid_idx[sel].astype(int)

            if cols.size > 0:
                edge_mean = np.asarray(self.edge_mean, dtype=np.float32)
                edge_std  = np.clip(np.asarray(self.edge_std, dtype=np.float32), 1e-8, None)
                if torch.is_tensor(e):
                    mean_t = torch.as_tensor(edge_mean[cols], dtype=e.dtype, device=e.device)
                    std_t  = torch.as_tensor(edge_std[cols],  dtype=e.dtype, device=e.device)
                    e[:, cols] = (e[:, cols] - mean_t) / std_t
                else:
                    e[:, cols] = (e[:, cols] - edge_mean[cols]) / edge_std[cols]

        return obj


    @staticmethod
    def _as_list(a):
        if a is None:
            return None
        if isinstance(a, (list, tuple, set)):
            seq = list(a)
        else:
            try:
                import numpy as np
                if isinstance(a, np.ndarray):
                    seq = a.ravel().tolist()
                else:
                    raise TypeError
            except Exception:
                try:
                    import torch
                    if torch.is_tensor(a):
                        seq = a.view(-1).tolist()
                    else:
                        seq = [a]
                except Exception:
                    seq = [a]
        return [int(x) for x in seq]


    def _load_stats(self, strict: bool) -> bool:
        # 1) pkl: train_mean_<tag>.pkl + train_std_<tag>.pkl
        try:
            if os.path.exists(self.stats_mean_path) and os.path.exists(self.stats_std_path):
                with open(self.stats_mean_path, "rb") as f:
                    mean_pack = pkl.load(f)
                with open(self.stats_std_path, "rb") as f:
                    std_pack = pkl.load(f)
                self.node_mean = np.asarray(mean_pack["node"], dtype=np.float32)
                self.edge_mean = np.asarray(mean_pack["edge"], dtype=np.float32)
                self.node_std  = np.clip(np.asarray(std_pack["node"], dtype=np.float32), 1e-8, None)
                self.edge_std  = np.clip(np.asarray(std_pack["edge"], dtype=np.float32), 1e-8, None)
                return True
        except Exception as e:
            if strict:
                raise
        if strict:
            raise FileNotFoundError(
                f"未找到训练统计：{self.stats_mean_path} + {self.stats_std_path} "
            )
        return False


    def _save_stats_files(self, node_mean, node_std, edge_mean, edge_std):
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.stats_mean_path, "wb") as f:
            pkl.dump({"node": node_mean.astype(np.float32), "edge": edge_mean.astype(np.float32)}, f)
        with open(self.stats_std_path, "wb") as f:
            pkl.dump({"node": node_std.astype(np.float32),  "edge": edge_std.astype(np.float32)},  f)
    
        print(f"✅ 训练统计已保存：\n  - {self.stats_mean_path}\n  - {self.stats_std_path}\n ")





