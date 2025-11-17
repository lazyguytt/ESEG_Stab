import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import defaultdict
import  re

#7
FEATURE_COLUMNS = [
    'blosum62',
    'hydropathy_diff',
    'polarity_diff',
    'aromatic_change',
    'delta_volume',
    'burial_prior',
    'rel_pos',
]

def align_features_to_vector(feat_dict, order=FEATURE_COLUMNS, fill=0.0):
    return np.asarray([float(feat_dict.get(k, fill)) for k in order], dtype=np.float32)

AA_PROPERTIES = {
    'A': {'hydropathy': 1.8, 'polarity': 8.1, 'charge': 0, 'mw': 89.1, 'aromatic': 0},
    'R': {'hydropathy': -4.5, 'polarity': 10.5, 'charge': 1, 'mw': 174.2, 'aromatic': 0},
    'N': {'hydropathy': -3.5, 'polarity': 11.6, 'charge': 0, 'mw': 132.1, 'aromatic': 0},
    'D': {'hydropathy': -3.5, 'polarity': 13.0, 'charge': -1, 'mw': 133.1, 'aromatic': 0},
    'C': {'hydropathy': 2.5, 'polarity': 5.5, 'charge': 0, 'mw': 121.2, 'aromatic': 0},
    'Q': {'hydropathy': -3.5, 'polarity': 10.5, 'charge': 0, 'mw': 146.2, 'aromatic': 0},
    'E': {'hydropathy': -3.5, 'polarity': 12.3, 'charge': -1, 'mw': 147.1, 'aromatic': 0},
    'G': {'hydropathy': -0.4, 'polarity': 9.0, 'charge': 0, 'mw': 75.1, 'aromatic': 0},
    'H': {'hydropathy': -3.2, 'polarity': 10.4, 'charge': 1, 'mw': 155.2, 'aromatic': 1},
    'I': {'hydropathy': 4.5, 'polarity': 5.2, 'charge': 0, 'mw': 131.2, 'aromatic': 0},
    'L': {'hydropathy': 3.8, 'polarity': 4.9, 'charge': 0, 'mw': 131.2, 'aromatic': 0},
    'K': {'hydropathy': -3.9, 'polarity': 11.3, 'charge': 1, 'mw': 146.2, 'aromatic': 0},
    'M': {'hydropathy': 1.9, 'polarity': 5.7, 'charge': 0, 'mw': 149.2, 'aromatic': 0},
    'F': {'hydropathy': 2.8, 'polarity': 5.2, 'charge': 0, 'mw': 165.2, 'aromatic': 1},
    'P': {'hydropathy': -1.6, 'polarity': 8.0, 'charge': 0, 'mw': 115.1, 'aromatic': 0},
    'S': {'hydropathy': -0.8, 'polarity': 9.2, 'charge': 0, 'mw': 105.1, 'aromatic': 0},
    'T': {'hydropathy': -0.7, 'polarity': 8.6, 'charge': 0, 'mw': 119.1, 'aromatic': 0},
    'W': {'hydropathy': -0.9, 'polarity': 5.4, 'charge': 0, 'mw': 204.2, 'aromatic': 1},
    'Y': {'hydropathy': -1.3, 'polarity': 6.2, 'charge': 0, 'mw': 181.2, 'aromatic': 1},
    'V': {'hydropathy': 4.2, 'polarity': 5.9, 'charge': 0, 'mw': 117.1, 'aromatic': 0}
}

AA_VOLUME = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,
             'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,
             'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}

BURIAL_PRIOR = {'W':0.9,'F':0.85,'Y':0.8,'I':0.8,'L':0.8,'V':0.75,
                'M':0.7,'C':0.65,'A':0.5,'T':0.5,'S':0.45,'G':0.45,
                'P':0.45,'H':0.55,'Q':0.55,'N':0.55,'K':0.4,'R':0.4,'E':0.35,'D':0.35}

_B62 = '''
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
'''

BLOSUM62 = {}

_lines = [ln for ln in _B62.strip().splitlines()]
cols = _lines[0].split()
for r in _lines[1:]:
    parts = r.split()
    aa = parts[0]
    for j, score in enumerate(parts[1:]):
        BLOSUM62[(aa, cols[j])] = int(score)

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
MUT_RE = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])")


def blosum62_score(wt_aa, mt_aa):
    if wt_aa is None or mt_aa is None: return 0
    return BLOSUM62.get((wt_aa, mt_aa), BLOSUM62.get((mt_aa, wt_aa), 0))


def parse_mutation(mutation: str):
    """解析标准突变格式 K28D: K 28 D"""
    if pd.isna(mutation): return None, None, None
    m = MUT_RE.fullmatch(str(mutation).strip().upper())
    if not m: return None, None, None
    wt_aa, pos, mt_aa = m.group(1), int(m.group(2)), m.group(3)
    return wt_aa, pos, mt_aa


def cheap_mutation_scores(wt_aa, mt_aa, pos, L):
    dvol = AA_VOLUME[mt_aa] - AA_VOLUME[wt_aa]
    rel_pos = pos / float(L)
    burial = BURIAL_PRIOR.get(wt_aa, 0.5)
    return {
        'delta_volume': float(dvol),
        'rel_pos': float(rel_pos),
        'burial_prior': float(burial),
    }


def calculate_local_features(wt_aa, mut_aa, pos, wt_seq, mt_seq):
    """
    - hydropathy_diff: wt - mut
    - polarity_diff: wt - mut
    - aromatic_change: 是否改变芳香性(0/1)
    """
    features = {}
    if (wt_aa not in AA_PROPERTIES or 
        mut_aa not in AA_PROPERTIES or 
        pos is None or 
        pos < 1 or 
        pos > len(wt_seq)):
        return features

    features.update({
        'hydropathy_diff': AA_PROPERTIES[wt_aa]['hydropathy'] - AA_PROPERTIES[mut_aa]['hydropathy'],
        'polarity_diff': AA_PROPERTIES[wt_aa]['polarity'] - AA_PROPERTIES[mut_aa]['polarity'],
        'aromatic_change': int(AA_PROPERTIES[wt_aa]['aromatic'] != AA_PROPERTIES[mut_aa]['aromatic']),
    })
    return features


def calculate_sequence_features(wt_seq, mt_seq, Mutation):
    features = defaultdict(float)
    wt_aa, pos, mut_aa = parse_mutation(Mutation)
    if any(v is None for v in [wt_aa, pos, mut_aa]):
        return dict(features)

    wt_seq, mt_seq = wt_seq.upper(), mt_seq.upper()
    L = len(wt_seq)

    features['blosum62'] = float(blosum62_score(wt_aa, mut_aa))

    features.update(calculate_local_features(wt_aa, mut_aa, pos, wt_seq, mt_seq))

    features.update(cheap_mutation_scores(wt_aa, mut_aa, pos, L))

    return dict(features)


def compute_normalization_parameters(feature_list):
    col_values = {k: [] for k in FEATURE_COLUMNS}
    for feat in feature_list:
        for k in FEATURE_COLUMNS:
            v = feat.get(k, 0.0)
            try:
                v = float(v)
            except Exception:
                v = 0.0
            if np.isnan(v) or np.isinf(v):
                v = 0.0
            col_values[k].append(v)

    mean_dict, std_dict = {}, {}
    for k in FEATURE_COLUMNS:
        arr = np.asarray(col_values[k], dtype=np.float64)
        if arr.size == 0:
            mean, std = 0.0, 1.0
        else:
            mean = float(arr.mean())
            std = float(arr.std(ddof=0))
            if std < 1e-8:
                std = 1e-8
        mean_dict[k] = mean
        std_dict[k] = std

    return mean_dict, std_dict


def normalize_features(feature_dict, mean_dict, std_dict):
    normalized = {}
    for k in FEATURE_COLUMNS:
        x = feature_dict.get(k, 0.0)
        try:
            x = float(x)
        except Exception:
            x = 0.0
        if np.isnan(x) or np.isinf(x):
            x = 0.0

        m = float(mean_dict.get(k, 0.0))
        s = float(std_dict.get(k, 1.0))
        if s < 1e-8:
            s = 1e-8
        normalized[k] = (x - m) / s
    return normalized
