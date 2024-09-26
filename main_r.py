#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:08:37 2024

"""

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import dgl
import os
import argparse
import yaml
import random
import statistics
import torch
import torch.distributed as dist




from logzero import logger
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from model.path_complex import PCNN
from rdkit.Chem import Descriptors
from torch.optim.lr_scheduler import StepLR
from utils.splitters import ScaffoldSplitter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
#from utils.path import path_complex_mol #previous
from utils.path_class import path_complex_mol
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

'''
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
'''
# 设置种子
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 使用一个具体的种子值



class CustomDataset(Dataset):
    def __init__(self, label_list,graph_feats, graph_list):
        self.labels = label_list
        self.graphs = graph_list
        self.graph_feats = graph_feats
        self.device = torch.device('cpu')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index].to(self.device)
        g_feats = self.graph_feats[index].to(self.device)
        graph1, graph2, graph3 = [graph.to(self.device) for graph in self.graphs[index]]
        graph_list = [graph1, graph2 ,graph3]
        return label,g_feats, graph_list



def collate_fn(batch):

    labels, graph_feat,graph_lists = map(list, zip(*batch))
    labels = torch.stack(labels)
    graph_feat = torch.stack(graph_feat)

    l_graphs = []
    g_graphs = []
    f_graphs = []
    for i in range(len(graph_lists)):
        l_graphs.append(graph_lists[i][0])
        g_graphs.append(graph_lists[i][1])
        f_graphs.append(graph_lists[i][2])

    l_g = dgl.batch(l_graphs)
    g_g = dgl.batch(g_graphs) 
    f_g = dgl.batch(f_graphs) 
    return labels, graph_feat,l_g, g_g, f_g





def mul_min_max_normalize(data):

    scaler = MinMaxScaler()
    # 找到最小值和最大值
    
    # 对每个数据点应用Min-Max归一化公式
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    min_val = scaler.data_min_
    max_val = scaler.data_max_
    scale_val = max_val - min_val

    return normalized_data, min_val, scale_val



def sig_min_max_normalize(data):
    # 找到最小值和最大值
    min_val = min(data)
    max_val = max(data)

    # 对每个数据点应用Min-Max归一化公式
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]

    return normalized_data, min_val, max_val


def mul_map(target,x,min_values,scale_values):
    #print(min_values)
    #print(scale_values)
    min_val = min_values[target]
    scale_val = scale_values[target]
    x_cpu = x.cpu().detach().numpy()
    #x_np = x_cpu.detach().numpy()
    original_data = x_cpu * scale_val + min_val
    return torch.tensor(original_data)



def inverse_min_max_normalize(x, min_val, max_val):
    # 对每个归一化后的数据点应用逆操作
    original_data = x * (max_val - min_val)
    return original_data



def is_file_in_directory(directory, target_file):
    file_path = os.path.join(directory, target_file)
    return os.path.isfile(file_path)


def get_qm9_task_names():
    """Get that default sider task names and return the side results for the drug"""
    '''
    ['E1-CC2','E2-CC2','f1-CC2','f2-CC2','E1-PBE0','E2-PBE0',
    'f1-PBE0','f2-PBE0','E1-CAM','E2-CAM','f1-CAM','f2-CAM']
    '''
    #return ['mu','alpha','homo','lumo','gap','r2','zpve','U0','U','H','G','Cv']
    return ['mu','alpha','homo','lumo','gap','r2','zpve','u0','u298','h298','g298','cv','u0_atom','u298_atom','h298_atom','g298_atom']

#u0,u298,h298,g298
    


def get_qm8_task_names():
    """Get that default sider task names and return the side results for the drug"""

    return ['E1-CC2','E2-CC2','f1-CC2','f2-CC2','E1-PBE0','E2-PBE0',
    'f1-PBE0','f2-PBE0','E1-CAM','E2-CAM','f1-CAM','f2-CAM']

def get_data_task_names():
    """Get that default sider task names and return the side results for the drug"""

    return ['label']



def global_feats(smiles):
    mol = Chem.MolFromSmiles(smiles)

    global_feat = []

    # 计算分子的质量
    mol_mass = Descriptors.ExactMolWt(mol)
    global_feat.extend([mol_mass])

    # 计算分子的表面积
    mol_area = Descriptors.TPSA(mol)
    global_feat.extend([mol_area])

    mol_with_hydrogens = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_hydrogens)
    volume = AllChem.ComputeMolVolume(mol_with_hydrogens)

    global_feat.extend([volume])


    # 计算分子的极性度
    mol_polarity = Descriptors.MolLogP(mol)
    global_feat.extend([mol_polarity])

    # 计算分子中不同类型的键的数量
    bond_feat = [0]*4
    num_bonds = mol.GetNumBonds()
    if num_bonds != 0:
        num_single_bonds = sum(bond.GetBondType() == Chem.BondType.SINGLE for bond in mol.GetBonds())
        bond_feat[0] = num_single_bonds/num_bonds
        num_double_bonds = sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in mol.GetBonds())
        bond_feat[1] = num_double_bonds/num_bonds
        num_triple_bonds = sum(bond.GetBondType() == Chem.BondType.TRIPLE for bond in mol.GetBonds())
        bond_feat[2] = num_triple_bonds/num_bonds
        num_aromtic_bonds = sum(bond.GetBondType() == Chem.BondType.AROMATIC for bond in mol.GetBonds())
        bond_feat[3] = num_aromtic_bonds/num_bonds

    global_feat.extend(bond_feat)

    return torch.tensor(global_feat)






def creat_data(datafile, encoder_atom,encoder_bond,encode_two_path,encode_tree_path,batch_size,train_ratio,vali_ratio,test_ratio):
    
    datasets = datafile
    directory_path = 'data/processed/'
    target_file_name = datafile +'.pth'


    if is_file_in_directory(directory_path, target_file_name):
        # 文件路径
        file_path = 'data/processed/'+ datasets+'.txt'

        # 从文件中读取包含两个数组的数据
        combined_data = np.loadtxt(file_path, delimiter='\t')

        if datasets in ['qm9','qm8']:
            # 切分数据为两个数组
            min_val = combined_data[:, 0]  # 第一列是 Array 1
            scale_val = combined_data[:, 1]  # 第二列是 Array 2
            return min_val, scale_val
        
        else:
            # 切分数据为两个数组
            min_val = combined_data[0]  # 第一列是 Array 1
            scale_val = combined_data[1]  # 第二列是 Array 2
            return min_val, scale_val


    
    else:
        df = pd.read_csv('data/' + datasets + '.csv')#
        HAR2EV = 27.2113825435  # 1 Hartree = 27.2114 eV 
        KCALMOL2EV = 0.04336414  # 1 Hartree = 27.2114 eV 
        if datasets == 'qm9':
            '''
            smiles_list = df['smiles']
            data = QM9Dataset(label_keys = get_qm9_task_names(), cutoff=5.0)
            # 使用示例
            labels_list = []

            for graph, labels in data:
                #all_labels.append(labels)
                labels_list.append(labels)

            df_labels = pd.DataFrame(labels_list, columns=get_qm9_task_names())
            '''
            smiles_list, df_labels = df['smiles'], df[get_qm9_task_names()]  
            

            labels, min_val, scale_val = mul_min_max_normalize(df_labels)

            #smiles_list, labels = df['smiles'], df[get_qm9_task_names()]  

        elif datasets == 'qm8':
            smiles_list, labels = df['smiles'], df[get_qm8_task_names()]  

            #min_val = labels.mean() 
            #scale_val = labels.std()
            #labels = (labels - min_val) / (scale_val + 1e-5)

            labels, min_val, scale_val = mul_min_max_normalize(labels)

        elif datasets in ['solv','lipo','esol','qm7']:
            smiles_list, labels = df['smiles'], df['label'] 
            labels, min_val, scale_val = sig_min_max_normalize(labels)


        data_list = []
        
        t = 0 
        for i in range(len(smiles_list)):
            smiles = smiles_list[i]
            #graph_feats = global_feats(smiles)
            graph_feats = torch.tensor([0])
            #if has_isolated_hydrogens(smiles) == False and conformers_is_zero(smiles) == True:
            #if conformers_is_zero(smiles) == True:

            Graph_list = path_complex_mol(smiles, encoder_atom,encoder_bond,encode_two_path,encode_tree_path)


            if i % 10000 == 0:
                print(i)
            
            if Graph_list != False:
                if datasets in ['solv','lipo','esol','qm7']:
                     data_list.append([smiles, torch.tensor(labels[i]), graph_feats, Graph_list])

                else:
                    data_list.append([smiles, torch.tensor(labels.iloc[i]), graph_feats, Graph_list])
            else:
                print(i)
                t += 1
        print('无效的个数：', t)

        #data_list = [['occr',albel,[c_size, features, edge_indexs],[g,liearn_g]],[],...,[]]

        print('Graph list was done!')

        splitter = ScaffoldSplitter().split(data_list, frac_train=train_ratio, frac_valid=vali_ratio, frac_test=test_ratio)
        
        print('splitter was done!')
        
        train_label = []
        train_graph_feat = []
        train_graph_list = []
        for tmp_train_graph in splitter[0]:
            
            train_label.append(tmp_train_graph[1])
            train_graph_feat.append(tmp_train_graph[2])
            train_graph_list.append(tmp_train_graph[3])

        valid_label = []
        valid_graph_feat = []
        valid_graph_list = []
        for tmp_valid_graph in splitter[1]:
            valid_label.append(tmp_valid_graph[1])
            valid_graph_feat.append(tmp_valid_graph[2])
            
            valid_graph_list.append(tmp_valid_graph[3])

        test_label = []
        test_graph_feat = []
        test_graph_list = []
        for tmp_test_graph in splitter[2]:
            test_label.append(tmp_test_graph[1])
            test_graph_feat.append(tmp_test_graph[2])
            test_graph_list.append(tmp_test_graph[3])

        #batch_size = 256

        torch.save({
            'train_label': train_label,
            'train_graph_feat':train_graph_feat,
            'train_graph_list': train_graph_list,
            'valid_label': valid_label,
            'valid_graph_feat': valid_graph_feat,
            'valid_graph_list': valid_graph_list,
            'test_label': test_label,
            'test_graph_feat': test_graph_feat,
            'test_graph_list': test_graph_list,
            'batch_size': batch_size,
            'shuffle': True,  # 保存时假设你在创建 DataLoader 时使用了 shuffle=True
            # 其他必要信息
        }, 'data/processed/'+ datafile +'.pth')

        # 文件路径
        file_path = 'data/processed/'+datasets+'.txt'

        # 将两个数组写入同一个文件，一次性写入
        combined_data = np.column_stack((min_val, scale_val))

        # 加入注释
        header = 'Array 1\tArray 2'
        np.savetxt(file_path, combined_data, header=header, delimiter='\t')
                    
        print('preparing in pytorch format!')
        return min_val, scale_val




class CustomLoss(nn.Module):
    def __init__(self, lambda_reg):
        super(CustomLoss, self).__init__()
        self.lambda_reg = lambda_reg  # 正则项的权重

    def forward(self, output, target, model):
        # 计算普通的损失（比如MSE损失）
        #criterion = nn.MSELoss()
        loss = loss_fn(output, target)
        
        # 加入L2范数的正则项
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)**2
        loss += self.lambda_reg * l2_reg
        
        return loss
    



def train(model, device, target, train_loader, valid_loader, optimizer, epoch, resent,pooling,datafile):
    model.train()

    total_train_loss = 0.0
    train_num = 0

    
    for labels, g_feats, g, lg, fg in train_loader:
        
        optimizer.zero_grad()
        
        g = g.to(device)
        lg = lg.to(device)
        fg = fg.to(device)
        #print(g,lg,fg)
        g_graph_node_feat = g.ndata['feat'].to(device)
        g_graph_edge_feat = g.edata['feat'].to(device)

        lg_graph_node_feat = lg.ndata['feat'].to(device)
        lg_graph_edge_feat = lg.edata['feat'].to(device)

        fg_graph_node_feat = fg.ndata['feat'].to(device)
        fg_graph_edge_feat = fg.edata['feat'].to(device)

        train_label_value = []
        #print(data[0].shape)
        if datafile in ['qm8','qm9']:
            y = labels[:,target]
        else:
            y = labels
        #y = labels[:,target]
        #y = labels
        train_label_value.append(torch.unsqueeze(y, dim=0))
        #graph_list = data[1]

        train_label_pridct_tensor = model(g_feats, g, lg, fg, g_graph_node_feat,g_graph_edge_feat, lg_graph_node_feat, lg_graph_edge_feat,fg_graph_node_feat,fg_graph_edge_feat, device = device, resent = resent,pooling=pooling)

        
        train_label_value_tensor = torch.cat(train_label_value, dim=0).to(device)
       # print(train_label_value_tensor.shape)

        train_label_pridct_tensor = torch.squeeze(train_label_pridct_tensor)
        train_label_value_tensor = torch.squeeze(train_label_value_tensor)

        output = train_label_pridct_tensor.float()
        label = train_label_value_tensor.float()

        train_loss = loss_fn(output, label)

        train_loss = torch.sum(train_loss)
        total_train_loss = total_train_loss + train_loss
        train_loss = train_loss.float()
        train_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=100.0)
        optimizer.step()
    
    # 在整个批次上进行一次梯度计算和裁剪
    '''
    model.eval()
    if isinstance(loaded_valid_loader, list):
        avg_vali_loss = 0
        
    else:
        total_loss_val = 0.0
        for labels, g_feats, g, lg, fg in valid_loader:
            
            g = g.to(device)
            lg = lg.to(device)
            fg = fg.to(device)

            g_graph_node_feat = g.ndata['feat'].to(device)
            g_graph_edge_feat = g.edata['feat'].to(device)

            lg_graph_node_feat = lg.ndata['feat'].to(device)
            lg_graph_edge_feat = lg.edata['feat'].to(device)

            fg_graph_node_feat = fg.ndata['feat'].to(device)
            fg_graph_edge_feat = fg.edata['feat'].to(device)


            label_value = []
            if datafile in ['qm8','qm9']:
                y = labels[:,target]
            else:
                y = labels
            #y = labels[:, target]
            #y = labels
            
            label_value.append(torch.unsqueeze(y, dim=0))
            #graph_list = data[1]

            label_pridct_tensor = model(g_feats, g, lg, fg, g_graph_node_feat,g_graph_edge_feat, lg_graph_node_feat, lg_graph_edge_feat,fg_graph_node_feat,fg_graph_edge_feat, device = device, resent = resent,pooling=pooling)
            
            label_value_tensor = torch.cat(label_value, dim=0).to(device)
            label_pridct_tensor = torch.squeeze(label_pridct_tensor)
            label_value_tensor = torch.squeeze(label_value_tensor)
            
            output = label_pridct_tensor.float()
            label = label_value_tensor.float()

            

            loss = loss_fn(output, label)

            loss = torch.sum(loss)
            total_loss_val += loss

    '''
    total_loss_val = 0.0
    print(f"Epoch {epoch}|Train Loss: {total_train_loss:.4f}|Average Vali Loss:{total_loss_val:.4f}")
    return total_train_loss, total_loss_val




def predicting(model, device,target, data_loader, min_val, max_val,resent,pooling,datafile,out_dim):
    model.eval()
    
    label_values = []
    total_preds = []

    if datafile  in ['qm8','qm9']:

        with torch.no_grad():
            for labels, g_feats, g, lg, fg in data_loader:

                g = g.to(device)
                lg = lg.to(device)
                fg = fg.to(device)
                
                g_graph_node_feat = g.ndata['feat'].to(device)
                g_graph_edge_feat = g.edata['feat'].to(device)

                lg_graph_node_feat = lg.ndata['feat'].to(device)
                lg_graph_edge_feat = lg.edata['feat'].to(device)

                fg_graph_node_feat = fg.ndata['feat'].to(device)
                fg_graph_edge_feat = fg.edata['feat'].to(device)
                
                y = labels[:,target]
                true = mul_map(target,y,min_val, max_val)
                #true = inverse_min_max_normalize(y,min_val, max_val)
                label_values.append(torch.unsqueeze(true, dim=0))
                #graph_list = data[1]
                
                output = model(g_feats, g, lg, fg, g_graph_node_feat,g_graph_edge_feat, lg_graph_node_feat, lg_graph_edge_feat,fg_graph_node_feat,fg_graph_edge_feat, device = device, resent = resent,pooling=pooling).cpu()

                #predict = inverse_min_max_normalize(output, min_val, max_val)
                predict = mul_map(target,output,min_val, max_val)
                total_preds.append(torch.unsqueeze(predict, dim=0))

            label_pred = torch.cat(total_preds, dim=0)
            label_true = torch.cat(label_values, dim=0)

    
    else:

        with torch.no_grad():
            for labels, g_feats,g, lg, fg in data_loader:
                
                g = g.to(device)
                lg = lg.to(device)
                fg = fg.to(device)

                g_graph_node_feat = g.ndata['feat'].to(device)
                g_graph_edge_feat = g.edata['feat'].to(device)

                lg_graph_node_feat = lg.ndata['feat'].to(device)
                lg_graph_edge_feat = lg.edata['feat'].to(device)

                fg_graph_node_feat = fg.ndata['feat'].to(device)
                fg_graph_edge_feat = fg.edata['feat'].to(device)
                
                
                y = labels
                #print(y.shape)
                true = inverse_min_max_normalize(y,min_val, max_val)
                
                label_values.append(torch.unsqueeze(true, dim=0))
                #graph_list = data[1]
                
                output = model(g_feats, g, lg, fg, g_graph_node_feat,g_graph_edge_feat, lg_graph_node_feat, lg_graph_edge_feat,fg_graph_node_feat,fg_graph_edge_feat, device = device, resent = resent,pooling=pooling).cpu()

                #predict = inverse_min_max_normalize(output, min_val, max_val)
                predict = inverse_min_max_normalize(output,min_val, max_val)
                total_preds.append(torch.unsqueeze(predict, dim=0))
    
                label_pred = torch.cat(total_preds, dim=0)
                label_true = torch.cat(label_values, dim=0)

    predict_value_tensor = label_pred.squeeze(dim=1)
    predict_value_tensor = predict_value_tensor.squeeze()

    #true_value_tensor = label_true.squeeze(dim=1)
    
    true_value_tensor = label_true.view(-1, out_dim)
    predict_value_tensor = predict_value_tensor.view(-1, out_dim)

    #print(true_value_tensor.shape)
    #print(predict_value_tensor.shape)
    if datafile in ['qm9','qm8','qm7']:
        #mae = mae_function(true_value_tensor, predict_value_tensor)
        mae = mean_absolute_error(true_value_tensor, predict_value_tensor)
        #mae = calc_mae(true_value_tensor, predict_value_tensor)
        return mae
    
    else:
        mse = mean_squared_error(true_value_tensor, predict_value_tensor)
        #mse = calc_rmse(true_value_tensor, predict_value_tensor)
        rmse = np.sqrt(mse)
        return rmse



def parse_arguments():
    parser = argparse.ArgumentParser(description="示例命令行工具")

    
    parser.add_argument("--config", type=str, help="配置文件路径")

    args = parser.parse_args()
    args.config = './config/r_path.yaml'
    
    if args.config:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)

        
        for key, value in config.items():
            setattr(args, key, value)

    return args





if __name__ == '__main__':
    
    #mp.set_start_method('spawn', force=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    
    # 设置种子
    seed = 42
    set_seed(seed)

    args = parse_arguments()
    for key, value in vars(args).items():
        if key != 'config':
            print(f"{key}: {value}")
    datafile = args.select_dataset
    '''
    if datafile in ['qm7','qm8','qm9']:
        batch_size = 256
    else:
        batch_size = 32
    '''
    batch_size = args.batch_size


    train_ratio = args.train_ratio
    vali_ratio = args.vali_ratio
    test_ratio = args.test_ratio

    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond
    encode_two_path = args.encode_two_path
    encode_tree_path = args.encode_tree_path

    encoder_atom_dim = {"basic":11,"cfid":438,"cgcnn":92,'self':62,'gme':47}
    encoder_bond_dim = {"dim_14":21,"dim_26":26}
    encode_two_path_dim = {"dim_8":9,"dim_10":10}
    encode_tree_path_dim ={"dim_6":7,"dim_15":15}


    encode_dim = [0,0,0,0]
    encode_dim[0] = encoder_atom_dim[encoder_atom]
    encode_dim[1] = encoder_bond_dim[encoder_bond]
    encode_dim[2] = encode_two_path_dim[encode_two_path]
    encode_dim[3] = encode_tree_path_dim[encode_tree_path]

    min_val, max_val = creat_data(datafile, encoder_atom, encoder_bond, encode_two_path,encode_tree_path, batch_size,train_ratio,vali_ratio,test_ratio)

    model_select = args.model_select
    resent = args.resent
    pooling = args.pooling
    loss_sclect = args.loss_sclect
    
    tras_med = args.tras_med

    if datafile == 'qm9':
        target = [2,3,4]
        
    elif datafile == 'qm8':
        target = [0,1,2,3,4,5,6,7,8,9,10,11]
        
    else:
        target = [0]

    out_dim = len(target)
    #out_dim = 1
    # 加载 DataLoader 使用的数据集和其他必要信息
    state = torch.load('data/processed/'+datafile+'.pth')

    # 重新创建 CustomDataset 和 DataLoader
    loaded_train_dataset = CustomDataset(state['train_label'],state['train_graph_feat'], state['train_graph_list'])
    loaded_valid_dataset = CustomDataset(state['valid_label'],state['valid_graph_feat'], state['valid_graph_list'])
    loaded_test_dataset = CustomDataset(state['test_label'], state['test_graph_feat'],state['test_graph_list'])
    
    print("length of training set:",len(loaded_train_dataset))
    print("length of validation set:",len(loaded_valid_dataset))
    print("length of testing set:",len(loaded_test_dataset))

    loaded_train_loader = DataLoader(loaded_train_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    if vali_ratio == 0.0:
        loaded_valid_loader = []
    else:
        loaded_valid_loader = DataLoader(loaded_valid_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    loaded_test_loader = DataLoader(loaded_test_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)


    print('dataset was loaded!')

    
    
    iter = args.iter
    LR = args.LR
    NUM_EPOCHS = args.NUM_EPOCHS
    num_blcok = args.num_blcok
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    num_heads = args.head
    num_layers = args.layer_number


    ALl_MAE = []

    

    if model_select == 'pcnn':
        model = PCNN(in_feats=10, hidden_size = 32, out_feats=64, encode_dim=encode_dim, out_dim = out_dim,tras_med = tras_med,num_blcok=num_blcok,num_heads=num_heads,num_layers=num_layers)

    else:
        print('No found model!!!')

    #print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    MAE_list = []
    train_loss_dic = {}
    vali_loss_dic = {}

    #model = nn.DataParallel(model)
    #model.to('cuda')
    model = model.to(device)
    if loss_sclect == 'l1':
        #loss_fn = nn.L1Loss(reduction='none')
        loss_fn = nn.L1Loss(reduction='sum')#sum,mean,none
        
    if loss_sclect == 'l2':
        loss_fn = nn.MSELoss(reduction='sum')

    if loss_sclect == 'sml1':
        loss_fn = nn.SmoothL1Loss(reduction='sum')#mean,none,sum

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_MAE = 1000
    for epoch in range(NUM_EPOCHS):
        train_loss,vali_loss = train(model, device, target, loaded_train_loader, loaded_valid_loader,optimizer, epoch + 1, resent, pooling, datafile)


        MAE = predicting(model, device, target, loaded_test_loader, min_val, max_val,resent, pooling,datafile, out_dim)
        
        
        if MAE < best_MAE:
            logger.info(f'MAE: {MAE:.5f}')
            formatted_number = "{:.5f}".format(MAE)
            best_MAE = float(formatted_number)
            MAE_list.append(best_MAE)
            print(f"Epoch [{epoch+1}], Learning Rate: {scheduler.get_last_lr()}")

        if epoch % 10 == 0:
            print("-------------------------------------------------------")
            print("epoch:",epoch)
            print('best_MAE:', best_MAE)
        
        if epoch == NUM_EPOCHS-1:
            #MAE_list.append(best_MAE)
            min_mae = min(MAE_list) 
            print(f"The best min mae score up to {i+1}-loop is :",min_mae)
            ALl_MAE.append(min_mae)
