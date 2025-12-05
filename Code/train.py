import os
import sys
import numpy as np
import datetime
import argparse
from tqdm import tqdm
from model import *
from dataio import *
from metrics import *
from generator import *
from losses import inv_log_cosh
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='EXAMPLE:\npython3 train.py -w best_model.pth -n 200 -c 64 -e 2 -d 8 -f 16 -p ../dl-training-data/ -v 0 -o /tmp/')
    parser.add_argument('-w', type=str, required=True, dest='file_weights', help="保存的参数文件")
    parser.add_argument('-n', type=int, required=True, dest='dev_size', help="用于培训的pdb数(全部使用-1)")
    parser.add_argument('-e', type=int, required=True, dest='training_epochs', help="# epochs数量")
    parser.add_argument('-o', type=str, required=True, dest='dir_out', help="目录，用于写入.npy文件")
    parser.add_argument('-p', type=str, required=True, dest='dir_dataset', help="所有数据(包括.lst)所在的路径")
    parser.add_argument('-v', type=int, required=True, dest='flag_eval_only', help="1 =只评估，不训练")
    args = parser.parse_args()
    return args

def load_model_state(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if torch.cuda.device_count() > 1:
            if not k.startswith('module.'):
                name = 'module.' + k  # 添加 'module.' 前缀
            else:
                name = k
        else:
            if k.startswith('module.'):
                name = k[7:]  # 去掉 'module.' 前缀
            else:
                name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)   
    
if torch.cuda.is_available():
    print("GPU is available.")
    device = torch.device('cuda')
else:
    print("GPU is not available, using CPU.")
    device = torch.device('cpu')
    
args = get_args()
file_weights = args.file_weights
dev_size = 9258
casp_size = 82
training_window = 128
training_epochs = args.training_epochs
arch_depth = 8
filters_per_layer = 16
dir_dataset = args.dir_dataset
dir_out = args.dir_out
flag_eval_only = args.flag_eval_only
pad_size = 10
batch_size = 2
expected_n_channels = 49

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

all_feat_paths = [
    os.path.join(dir_dataset, 'deepcov', 'features/'),
    os.path.join(dir_dataset, 'CASP13', 'features/'),
    os.path.join(dir_dataset, 'CASP14', 'features/'),
    os.path.join(dir_dataset, 'CASP15', 'features/'),
    os.path.join(dir_dataset, 'CASP', 'features/'),
    os.path.join(dir_dataset, 'cameo', 'features/')
]
all_dist_paths = [
    os.path.join(dir_dataset, 'deepcov', 'distance/'),
    os.path.join(dir_dataset, 'CASP13', 'distance/'),
    os.path.join(dir_dataset, 'CASP14', 'distance/'),
    os.path.join(dir_dataset, 'CASP15', 'distance/'),
    os.path.join(dir_dataset, 'CASP', 'distance/'),
    os.path.join(dir_dataset, 'cameo', 'distance/')
]

model = Resmambakan(expected_n_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = inv_log_cosh


if not flag_eval_only: 
    deepcov_list = load_list(dir_dataset + '/deepcov.lst', dev_size)
    casp_list=load_list(dir_dataset + '/CASP.lst', casp_size)

    length_dict = {}
    (ly, seqy, cb_map) = np.load(dir_dataset + '/deepcov/distance/' + pdb + '-cb.npy', allow_pickle=True)
    length_dict[pdb] = ly
    
    casp_dict = {}
    (ly, seqy, cb_map) = np.load(dir_dataset + '/CASP/distance/' + pdbc + '-cb.npy', allow_pickle=True)
    casp_dict[pdbc] = ly

    train_pdbs = deepcov_list
    valid_pdbs = casp_list

    train_dataset = DistDataset(train_pdbs, all_feat_paths, all_dist_paths, training_window, pad_size, expected_n_channels, label_engineering='16.0')
    valid_dataset = DistDataset(valid_pdbs, all_feat_paths, all_dist_paths, training_window, pad_size, expected_n_channels, label_engineering='16.0')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    if os.path.exists(file_weights):
        print('')
        print('加载存在的权重-Loading existing weights..')
        model.load_state_dict(torch.load(file_weights), strict=False)

    print('')
    print('训练-Train..')

    best_val_loss = float('inf')
    best_val_mae = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }

    for epoch in range(training_epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{training_epochs}', unit='batch', position=0,
                  leave=True) as pbar:
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, Y_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                output_np = output.detach().cpu().numpy()
                Y_batch_np = Y_batch.detach().cpu().numpy()

                output_np_flat = output_np.reshape(output_np.shape[0], -1)
                Y_batch_np_flat = Y_batch_np.reshape(Y_batch_np.shape[0], -1)

                mae = mean_absolute_error(Y_batch_np_flat, output_np_flat)

                train_loss += loss.item()
                train_mae += mae.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{mae.item():.4f}'})
                pbar.update(1)

        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in valid_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, Y_batch)

                output_np = output.detach().cpu().numpy()
                Y_batch_np = Y_batch.detach().cpu().numpy()

                output_np_flat = output_np.reshape(output_np.shape[0], -1)
                Y_batch_np_flat = Y_batch_np.reshape(Y_batch_np.shape[0], -1)

                mae = mean_absolute_error(Y_batch_np_flat, output_np_flat)
                val_loss += loss.item()
                val_mae += mae.item()
                

        val_loss /= len(valid_loader)
        val_mae /= len(valid_loader)
        
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        print(
            f'Epoch {epoch + 1}/{training_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), file_weights)
            print(f'Best model saved at epoch {epoch + 1} with validation loss {val_loss:.4f}')
else:
    print('只评估模式...')
    if os.path.exists(file_weights):
        load_model_state(model, file_weights)
    else:
        print('No weights file found. Exiting.')
        sys.exit(1)
        
    psicov_list = load_list(os.path.join(dir_dataset, 'CASP13.lst'))
    psicov_length_dict = {}
    valid_pdbs = []
    for pdb in psicov_list:
        try:
            data = np.load(os.path.join(dir_dataset, 'CASP13/distance', f"{pdb}-cb.npy"), allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                ly, seqy, cb_map = data
            else:
                cb_map = data
                ly = cb_map.shape[0]
                
            psicov_length_dict[pdb] = ly
            valid_pdbs.append(pdb)
        except Exception as e:
            print(f"Error loading {pdb}: {str(e)}")
            continue
    
    psicov_list = valid_pdbs   
    evalsets = {'psicov': {'LMAX': 400, 'list': psicov_list, 'lendict': psicov_length_dict}}

    def evaluate_model(evalsets, file_weights, arch_depth, filters_per_layer, expected_n_channels, pad_size, all_feat_paths, all_dist_paths, device, flag_plots=False, dir_out='./'):
        for my_eval_set in evalsets:
            print(f'Evaluate on the {my_eval_set} set..')
            my_list = evalsets[my_eval_set]['list']
            LMAX = evalsets[my_eval_set]['LMAX']
            length_dict = evalsets[my_eval_set]['lendict']
            valid_samples = []
            for pdb in my_list:
                try:
                    data = np.load(os.path.join(dir_dataset, 'CASP13/distance', f"{pdb}-cb.npy"), allow_pickle=True)
                    if isinstance(data, np.ndarray) and data.dtype == object:
                        ly, seqy, cb_map = data
                    else:
                        cb_map = data
                        ly = cb_map.shape[0]
                
                
                    if np.isnan(cb_map).any():
                        print(f"NaN in distance map: {pdb}")
                        continue
                
                    valid_samples.append(pdb)
                    length_dict[pdb] = ly 
                except Exception as e:
                    print(f"Error in evaluation loader: {pdb} - {str(e)}")
                    continue
            my_list = valid_samples
            model = Resmambakan(expected_n_channels).to(device)
            model = model.to(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            load_model_state(model, file_weights)
            model.eval()
            
            my_loader = DataLoader(DistDataset(my_list, all_feat_paths, all_dist_paths, LMAX, pad_size, expected_n_channels), batch_size=1, shuffle=False)

            predictions, targets = [], []
            with torch.no_grad():
                for X_batch, Y_batch in my_loader:
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                    output = model(X_batch)
                    predictions.append(output.cpu().numpy())
                    targets.append(Y_batch.cpu().numpy())

                    del X_batch, Y_batch, output
                    torch.cuda.empty_cache()

            P = np.concatenate(predictions, axis=0)
            Y = np.full((len(my_loader), LMAX, LMAX, 1), np.nan)

            for i, (X_batch, Y_batch) in enumerate(my_loader):
                Y_batch_np = Y_batch.cpu().numpy()
                Y[i, :, :, 0] = Y_batch_np[0, 0, :, :]

            for j in range(0, len(P[0, 0, :, 0])):
                for k in range(j, len(P[0, 0, :, 0])):
                    P[:, :, j, k] = (P[:, :, k, j] + P[:, :, j, k]) / 2.0
            P[P < 0.01] = 0.01

            Y = np.transpose(Y, (0, 3, 1, 2))

            P_2d = P[:, 0, :, :]
            Y_2d = Y[:, 0, :, :]

            results_list = evaluate_distances(P, Y, my_list, length_dict)
            print('计算各列的平均值，并打印结果')
            numcols = len(results_list[0].split())
            print(f'Averages for {my_eval_set}', end=' ')
            for i in range(2, numcols):
                x = results_list[0].split()[i].strip()
                if x == 'count' or results_list[0].split()[i - 1].strip() == 'count':
                    continue
                avg_this_col = False
                if x == 'nan':
                    avg_this_col = True
                try:
                    float(x)
                    avg_this_col = True
                except ValueError:
                    None
                if not avg_this_col:
                    print(x, end=' ')
                    continue
                avg = 0.0
                count = 0
                for mrow in results_list:
                    a = mrow.split()
                    if len(a) != numcols:
                        continue
                    x = a[i]
                    if x == 'nan':
                        continue
                    try:
                        avg += float(x)
                        count += 1
                    except ValueError:
                        print(f'ERROR!! float value expected!! {x}')
                if count == 0:
                    print(f'No valid items for {results_list[0].split()[i]}', end=' ')
                else:
                    print(f'AVG: {avg / count:.4f} items={count}', end=' ')
                    
            print('')
            print('Save predictions..')
            save_dir = os.path.join('results', dir_out)
            print(f"Saving predictions to: {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            for i in range(len(my_list)):
                L = length_dict[my_list[i]]
                save_path = os.path.join(save_dir, f"{my_list[i]}.npy")
                np.save(save_path, P[i, 0, :L, :L])
              
    evaluate_model(evalsets, file_weights, arch_depth, filters_per_layer, expected_n_channels, pad_size, all_feat_paths, all_dist_paths, device=device, flag_plots=False, dir_out=dir_out)
