import pickle
import random
import os
import numpy as np
import torch
from sklearn.decomposition import PCA

# 从指定的文件中读取蛋白质列表，并可限制最大读取数量。这通常用于加载一组需要进一步分析或处理的蛋白质ID。
def load_list(file_lst, max_items=1000000):
    if max_items < 0:
        max_items = 1000000
    protein_list = []
    with open(file_lst, 'r') as f:
        for l in f.readlines():
            protein_list.append(l.strip().split()[0])
    if max_items < len(protein_list):
        protein_list = protein_list[:max_items]
    return protein_list

# 统计和打印输入特征（例如某种类型的蛋白质特征矩阵）的通道（或层）的平均值、最大值和总和。
def summarize_channels(x, y):
    print('打印输入特征')
    print(' 通道数        平均值        最大值        总和')
    #print(' Channel        Avg        Max        Sum')
    for i in range(x.shape[2]):
        (m, s, a) = (x[:, :, i].flatten().max(), x[:, :, i].flatten().sum(), x[:, :, i].flatten().mean())
        print(' %7s %10.4f %10.4f %10.1f' % (i + 1, a, m, s))
    print("      Ymin = %.2f  Ymean = %.2f  Ymax = %.2f" % (y.min(), y.mean(), y.max()))

#函数直接获取并返回原始的距离矩阵。获取输入特征和分箱后的输出距离图。对输入和输出数据进行填充和裁剪，使其符合预期的尺寸。
def get_input_output_dist(pdb_id_list, all_feat_paths, all_dist_paths, pad_size, OUTL, expected_n_channels):
    XX = np.full((len(pdb_id_list), OUTL, OUTL, expected_n_channels), 0.0)
    #get_input_output_dist 中的输出矩阵 YY 维度为 (len(pdb_id_list), OUTL, OUTL, 1)，表示每个距离值。距离矩阵 Y 的填充值为 100.0。
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), 100.0)
    for i, pdb in enumerate(pdb_id_list):
        X = get_feature(pdb, all_feat_paths, expected_n_channels)
        assert X.shape[2] == expected_n_channels
        #直接使用 get_map 函数获取距离矩阵 Y。
        Y0 = get_map(pdb, all_dist_paths, X.shape[0])
        assert X.shape[0] >= Y0.shape[0]
        # if X.shape[0] != Y0.shape[0]:
        #     print()
            #print('\nWARNING!! Different len(X) and len(Y) for', pdb, X.shape[0], Y0.shape[0])
        l = X.shape[0]
        Y = np.full((l, l), np.nan)
        Y[:Y0.shape[0], :Y0.shape[0]] = Y0
        Xpadded = np.zeros((l + pad_size, l + pad_size, X.shape[2]), dtype=np.float32)
        Xpadded[int(pad_size / 2): l + int(pad_size / 2), int(pad_size / 2): l + int(pad_size / 2), :] = X
        Ypadded = np.full((l + pad_size, l + pad_size), 100.0, dtype=np.float32)
        Ypadded[int(pad_size / 2): l + int(pad_size / 2), int(pad_size / 2): l + int(pad_size / 2)] = Y
        l = Xpadded.shape[0]
        if l <= OUTL:
            XX[i, :l, :l, :] = Xpadded
            YY[i, :l, :l, 0] = Ypadded
        else:
            rx = random.randint(0, l - OUTL)
            ry = random.randint(0, l - OUTL)
            assert rx + OUTL <= l
            assert ry + OUTL <= l
            XX[i, :, :, :] = Xpadded[rx:rx + OUTL, ry:ry + OUTL, :]
            YY[i, :, :, 0] = Ypadded[rx:rx + OUTL, ry:ry + OUTL]
    return torch.tensor(XX, dtype=torch.float32), torch.tensor(YY, dtype=torch.float32)

def get_feature(pdb, all_feat_paths, expected_n_channels):
    features = None
    for path in all_feat_paths:
        if os.path.exists(path + pdb + '.pkl'):
            features = pickle.load(open(path + pdb + '.pkl', 'rb'))
    if features is None:
        print('期待的特征文件', pdb, '在这里没有找到', all_feat_paths)
        #print('Expected feature file for', pdb, 'not found at', all_feat_paths)
        exit(1)
    l = len(features['seq'])
    seq = features['seq']
    # Create X and Y placeholders
    X = np.full((l, l, expected_n_channels), 0.0)
    fi = 0
    # Add PSSM
    pssm = features['pssm']
    assert pssm.shape == (l, 22)
    for j in range(22):
        a = np.repeat(pssm[:, j].reshape(1, l), l, axis=0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add entropy
    entropy = features['entropy']
    assert entropy.shape == (l, )
    a = np.repeat(entropy.reshape(1, l), l, axis=0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add CCMpred
    ccmpred = features['ccmpred']
    assert ccmpred.shape == (l, l)
    X[:, :, fi] = ccmpred
    fi += 1
    # Add FreeContact
    freecon = features['freecon']
    assert freecon.shape == (l, l)
    X[:, :, fi] = freecon
    fi += 1
    # Add potential
    potential = features['potential']
    assert potential.shape == (l, l)
    X[:, :, fi] = potential
    fi += 1
    assert fi == expected_n_channels
    assert X.max() < 100.0
    assert X.min() > -100.0
    return X
def get_map(pdb, all_dist_paths, expected_l=-1):
    seqy = None
    cb_map = None
    ly = None
    mypath = ''
    for path in all_dist_paths:
        file_path = os.path.join(path, pdb + '-cb.npy')
        if os.path.exists(file_path):
            try:
                data = np.load(file_path, allow_pickle=True)
                mypath = file_path
                if isinstance(data, np.ndarray) and data.dtype == object and len(data) == 3:
                    ly, seqy, cb_map = data
                elif isinstance(data, np.ndarray) and data.ndim == 2:
                    cb_map = data
                    ly = cb_map.shape[0]
                    seqy = ""  # 空序列标记
                else:
                    raise ValueError(f"Unsupported data format in {file_path}")
                break
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
    if cb_map is None:
        raise FileNotFoundError(f"Distance map for {pdb} not found in: {all_dist_paths}")
    if 'cameo' not in mypath and expected_l > 0:
        expected_l = int(expected_l)
        ly = int(cb_map.shape[0])
        if ly != expected_l:
            if ly < expected_l:
                pad = expected_l - ly
                cb_map = np.pad(cb_map, ((0, pad), (0, pad)), mode='constant', constant_values=1.0)
            else:
                cb_map = cb_map[:expected_l, :expected_l]
            ly = expected_l
    Y = cb_map.copy()
    if 'cameo' not in mypath:
        assert not np.any(np.isnan(Y)), f"NaN values found in non-cameo map: {mypath}"
    if np.any(np.isnan(Y)):
        np.seterr(invalid='ignore')
        print(f'\nWARNING!! NaN values in {pdb} (l={ly}) at indices: {np.where(np.isnan(np.diagonal(Y)))}')
    Y[Y < 1.0] = 1.0
    Y[0, 0] = Y[0, 1] if not np.isnan(Y[0, 1]) else 1.0
    Y[ly - 1, ly - 1] = Y[ly - 1, ly - 2] if not np.isnan(Y[ly - 1, ly - 2]) else 1.0
    for q in range(1, ly - 1):
        if np.isnan(Y[q, q]):
            continue
        if np.isnan(Y[q, q - 1]) and np.isnan(Y[q, q + 1]):
            Y[q, q] = 1.0
        elif np.isnan(Y[q, q - 1]):
            Y[q, q] = Y[q, q + 1] if not np.isnan(Y[q, q + 1]) else 1.0
        elif np.isnan(Y[q, q + 1]):
            Y[q, q] = Y[q, q - 1] if not np.isnan(Y[q, q - 1]) else 1.0
        else:
            Y[q, q] = (Y[q, q - 1] + Y[q, q + 1]) / 2.0
    assert np.nanmax(Y) <= 500.0, f"Distance value exceeds 500 in {pdb}"
    assert np.nanmin(Y) >= 1.0, f"Distance value below 1.0 in {pdb}"
    return Y