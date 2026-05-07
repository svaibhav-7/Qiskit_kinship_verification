import os
import numpy as np
import scipy.io
from PIL import Image
from .quantum_core import compress_embedding

def load_kinfacew(root, cnn_extractor, n_qubits, max_pairs=None):
    """Load KinFaceW-I or KinFaceW-II using .mat metadata."""
    relations = ['fd', 'fs', 'md', 'ms']
    rel_dirs = {'fd': 'father-dau', 'fs': 'father-son',
                'md': 'mother-dau', 'ms': 'mother-son'}
    X1, X2, Y = [], [], []
    skipped = 0

    for rel in relations:
        mat_path = os.path.join(root, "meta_data", f"{rel}_pairs.mat")
        if not os.path.exists(mat_path):
            continue
        data = scipy.io.loadmat(mat_path)
        pairs = data['pairs']
        img_dir = os.path.join(root, "images", rel_dirs[rel])
        count = 0
        for row in pairs:
            label = int(row[1].flat[0])    # 1=kin, 0=non-kin
            img1 = str(row[2].flat[0])
            img2 = str(row[3].flat[0])
            p1 = os.path.join(img_dir, img1)
            p2 = os.path.join(img_dir, img2)
            if not os.path.exists(p1) or not os.path.exists(p2):
                skipped += 1
                continue
            
            e1 = cnn_extractor.extract(p1)
            e2 = cnn_extractor.extract(p2)
            X1.append(compress_embedding(e1, n_qubits))
            X2.append(compress_embedding(e2, n_qubits))
            Y.append(+1.0 if label == 1 else -1.0)
            count += 1
            if max_pairs and count >= max_pairs:
                break
    return np.array(X1), np.array(X2), np.array(Y)

def load_tskinface(root, cnn_extractor, n_qubits, max_families=200):
    """Load TSKinFace — generate parent-child kin pairs + non-kin pairs."""
    X1, X2, Y = [], [], []
    kin_embs = []

    for folder in ['FMS', 'FMD']:
        fdir = os.path.join(root, folder)
        if not os.path.exists(fdir):
            continue
        child_key = 'S' if folder == 'FMS' else 'D'
        families = set()
        for f in os.listdir(fdir):
            if f.endswith('.jpg'):
                parts = f.replace('.jpg','').split('-')
                if len(parts) == 3:
                    families.add(int(parts[1]))
        families = sorted(families)[:max_families]
        for fid in families:
            f_path = os.path.join(fdir, f"{folder}-{fid}-F.jpg")
            m_path = os.path.join(fdir, f"{folder}-{fid}-M.jpg")
            c_path = os.path.join(fdir, f"{folder}-{fid}-{child_key}.jpg")
            if not all(os.path.exists(p) for p in [f_path, m_path, c_path]):
                continue
            ef = cnn_extractor.extract(f_path)
            em = cnn_extractor.extract(m_path)
            ec = cnn_extractor.extract(c_path)
            
            X1.append(compress_embedding(ef, n_qubits)); X2.append(compress_embedding(ec, n_qubits)); Y.append(+1.0)
            X1.append(compress_embedding(em, n_qubits)); X2.append(compress_embedding(ec, n_qubits)); Y.append(+1.0)
            kin_embs.extend([ef, em, ec])

    # Generate non-kin pairs
    n_kin = sum(1 for y in Y if y == 1.0)
    rng = np.random.default_rng(42)
    n_embs = len(kin_embs)
    added = 0
    while added < n_kin and n_embs > 1:
        i, j = rng.choice(n_embs, size=2, replace=False)
        X1.append(compress_embedding(kin_embs[i], n_qubits))
        X2.append(compress_embedding(kin_embs[j], n_qubits))
        Y.append(-1.0)
        added += 1

    return np.array(X1), np.array(X2), np.array(Y)
