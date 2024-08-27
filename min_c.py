#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:08:37 2024

@author: longlee
"""

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
import random
import dgl
import statistics


from logzero import logger
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score,r2_score,mean_squared_error,mean_absolute_error
from sklearn.metrics import precision_recall_curve, auc

from sklearn import metrics
from model.path_complex import PCNN
from torch.optim.lr_scheduler import StepLR
from ruamel.yaml import YAML
from utils.splitters import ScaffoldSplitter
from utils.path_class import path_complex_mol
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

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



def has_node_with_zero_in_degree(graph):
    if (graph.in_degrees() == 0).any():
                return True
    
    #for graph in graph_list:
    #    if (graph.in_degrees() == 0).any():
    #        return True
        
    return False


def has_isolated_hydrogens(samiles):
    # 获取分子中的原子
    molecule = Chem.MolFromSmiles(samiles)
    mol = Chem.AddHs(molecule)  # 加氢
    if molecule is None:
        return True
    
    atoms = mol.GetAtoms()
    if len(atoms) <= 2:
        return True
    
    # 遍历原子
    for atom in atoms:
        # 如果原子是氢原子且没有邻居
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0:
            return True  # 存在孤立的氢原子
    
    return False  # 不存在孤立的氢原子





def conformers_is_zero(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # 加氢
    AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42) 
    # 检查是否有构象
    num_conformers = mol.GetNumConformers()

    G = nx.Graph()
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # 检查图是否为连通图
    if G.number_of_nodes() > 0:
        is_connected = nx.is_connected(G)
        if num_conformers > 0 and is_connected == True:
            return True
    else:
        return False
    

    
def min_max_normalize(data):
    # 找到最小值和最大值
    min_val = min(data)
    max_val = max(data)

    # 对每个数据点应用Min-Max归一化公式
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]

    return normalized_data, min_val, max_val


def inverse_min_max_normalize(x, min_val, max_val):
    # 对每个归一化后的数据点应用逆操作
    original_data = x * (max_val - min_val)
    return original_data

def is_file_in_directory(directory, target_file):
    file_path = os.path.join(directory, target_file)
    return os.path.isfile(file_path)


def unique(class_target):
    # 假设 y_true_np 是你的 NumPy 数组
    unique_classes, counts = np.unique(class_target, return_counts=True)

    # 打印唯一的类别和它们的出现次数
    for class_label, count in zip(unique_classes, counts):
        print(f"Class {class_label}: {count} samples")

    # 检查类别数量
    num_classes = len(unique_classes)
    if num_classes == 2:
        print("y_true_np 包含两个不同的类别.")
    else:
        print("y_true_np 不包含两个不同的类别.")

#others
def get_label():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['label']


#tox21,12     
def get_tox():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

#clintox,2
def get_clintox():
    
    return ['FDA_APPROVED', 'CT_TOX']

#sider,27
def get_sider():

    return ['Hepatobiliary disorders',
           'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
           'Investigations', 'Musculoskeletal and connective tissue disorders',
           'Gastrointestinal disorders', 'Social circumstances',
           'Immune system disorders', 'Reproductive system and breast disorders',
           'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
           'General disorders and administration site conditions',
           'Endocrine disorders', 'Surgical and medical procedures',
           'Vascular disorders', 'Blood and lymphatic system disorders',
           'Skin and subcutaneous tissue disorders',
           'Congenital, familial and genetic disorders',
           'Infections and infestations',
           'Respiratory, thoracic and mediastinal disorders',
           'Psychiatric disorders', 'Renal and urinary disorders',
           'Pregnancy, puerperium and perinatal conditions',
           'Ear and labyrinth disorders', 'Cardiac disorders',
           'Nervous system disorders',
           'Injury, poisoning and procedural complications']

#muv
def get_muv():
    
    return ['MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692',
            'MUV-712','MUV-713','MUV-733','MUV-737','MUV-810','MUV-832','MUV-846',
            'MUV-852',	'MUV-858','MUV-859']


def auc_function(y_true, y_pred):
    """
    计算两个张量之间的均方根误差（RMSE）。

    参数:
    - y_true (torch.Tensor): 真实标签的张量。
    - y_pred (torch.Tensor): 预测值的张量。

    返回:
    - torch.Tensor: RMSE 值。
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    y_true = y_true.to(torch.float32)
    assert y_true.dtype == y_pred.dtype, "y_true and y_pred must have the same dtype"
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    unique(y_true_np)
    unique(y_pred_np)

    auc = roc_auc_score(y_true_np, y_pred_np)
    auc = auc.item()
    return auc



def creat_data(datafile, encoder_atom, encoder_bond,encode_two_path, encode_tree_path, batch_size,train_ratio,vali_ratio,test_ratio):
    

    datasets = datafile

    directory_path = 'data/processed/'
    target_file_name = datafile +'.pth'

    if is_file_in_directory(directory_path, target_file_name):

        return True
    
    else:

        df = pd.read_csv('data/' + datasets + '.csv')#
        if datasets == 'tox21':
            smiles_list, labels = df['smiles'], df[get_tox()] 
            #labels = labels.replace(0, -1)
            labels = labels.fillna(0)

        if datasets == 'muv':
            smiles_list, labels = df['smiles'], df[get_muv()]  
            labels = labels.fillna(0)

        if datasets == 'sider':
            smiles_list, labels = df['smiles'], df[get_sider()]  

        if datasets == 'clintox':
            smiles_list, labels = df['smiles'], df[get_clintox()] 
    

        if datasets in ['hiv','bbbp','bace']:
            smiles_list, labels = df['smiles'], df[get_label()] 
            
        #labels = labels.replace(0, -1)
        #labels = labels.fillna(0)

        #smiles_list, labels = df['smiles'], df['label']        
        #labels = labels.replace(0, -1)
        
        #labels, min_val, max_val = min_max_normalize(labels)

        data_list = []
        feature_sets = ("atomic_number", "basic", "cfid", "cgcnn")
        for i in range(len(smiles_list)):
            if i % 1000 == 0:
                print(i)

            smiles = smiles_list[i]
            graph_feats = torch.tensor([0])
            #if has_isolated_hydrogens(smiles) == False and conformers_is_zero(smiles) == True :

            Graph_list = path_complex_mol(smiles, encoder_atom, encoder_bond,encode_two_path, encode_tree_path)
            if Graph_list == False:
                continue
            
            else:
                data_list.append([smiles, torch.tensor(labels.iloc[i]),graph_feats,Graph_list])



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
        



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets):
        logits = self.bce_with_logits(inputs, targets)
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        return torch.mean(alpha_factor * modulating_factor * logits)



def train(model, device, train_loader, valid_loader, optimizer, epoch):
    model.train()

    total_train_loss = 0.0
    train_num = 0

    for y, g_feats, g, lg, fg in train_loader:
        
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
        '''
        print("Max of g_graph_node_feat:", g_graph_node_feat.max().item())

        print("Max of g_graph_edge_feat:", g_graph_edge_feat.max().item())
        print("Min of g_graph_edge_feat:", g_graph_edge_feat.min().item())

        print("Max of lg_graph_edge_feat:", lg_graph_edge_feat.max().item())
        print("Min of lg_graph_edge_feat:", lg_graph_edge_feat.min().item())

        print("Max of fg_graph_edge_feat:", fg_graph_edge_feat.max().item())
        print("Min of fg_graph_edge_feat:", fg_graph_edge_feat.min().item())
        '''
        output = model(g_feats, g, lg, fg, g_graph_node_feat,g_graph_edge_feat, lg_graph_node_feat, lg_graph_edge_feat,fg_graph_node_feat,fg_graph_edge_feat, device = device, resent = resent,pooling=pooling).cpu()
        

        arr_label = torch.Tensor().cpu()
        arr_pred = torch.Tensor().cpu()
        for j in range(y.shape[1]):
            c_valid = np.ones_like(y[:, j], dtype=bool)
            c_label, c_pred = y[c_valid, j], output[c_valid, j]
            zero = torch.zeros_like(c_label)
            c_label = torch.where(c_label == -1, zero, c_label)
            
            arr_label = torch.cat((arr_label,c_label),0)
            arr_pred = torch.cat((arr_pred,c_pred),0)
        
        arr_pred = arr_pred.float()
        arr_label = arr_label.float()
        
        #print("Max of arr_pred:", arr_pred.max().item())
        #print("Min of arr_pred:", arr_pred.min().item())

        loss = loss_fn(arr_pred, arr_label)
        
        train_loss = torch.sum(loss)
        total_train_loss = total_train_loss + train_loss
        train_loss.backward()
        optimizer.step()
    
    # 在整个批次上进行一次梯度计算和裁剪
    '''
    if isinstance(loaded_valid_loader, list):
        avg_vali_loss = 0
    else:
        model.eval()
        total_loss_val = 0.0
        vali_num = 0
        arr_data = []

        for batch_idx, data in enumerate(valid_loader):

            label_value = []
            y = data[0]
            label_value.append(torch.unsqueeze(y, dim=0))
            graph_list = update_node_features(data[1]).to(device)
            node_features = graph_list.ndata['feat'].to(device)
            #output = model(batch_g_list = graph_list, device = device, resent = resent,pooling=pooling).cpu()
            output = model(graph_list, node_features).cpu()

            
            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()
            for j in range(y.shape[1]):
                c_valid = np.ones_like(y[:, j], dtype=bool)
                c_label, c_pred = y[c_valid, j], output[c_valid, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)
                
                arr_label = torch.cat((arr_label,c_label),0)
                arr_pred = torch.cat((arr_pred,c_pred),0)
            
            arr_pred = arr_pred.float()
            arr_label = arr_label.float()
            loss = loss_fn(arr_pred, arr_label)
            #loss = FocalLoss(arr_pred, arr_label)

            loss = torch.sum(loss)
            total_loss_val += loss
        '''
    total_loss_val = 0.0
    print(f"Epoch {epoch}|Train Loss: {total_train_loss:.4f}| Vali Loss:{total_loss_val:.4f}")

    return total_train_loss, total_loss_val


def predicting(model, device, data_loader):
    model.eval()
    
    total_preds = torch.Tensor().cpu()
    total_labels = torch.Tensor().cpu()

    with torch.no_grad():
        for y, g_feats, g, lg, fg in data_loader:
            
            g = g.to(device)
            lg = lg.to(device)
            fg = fg.to(device)
            
            g_graph_node_feat = g.ndata['feat'].to(device)
            g_graph_edge_feat = g.edata['feat'].to(device)

            lg_graph_node_feat = lg.ndata['feat'].to(device)
            lg_graph_edge_feat = lg.edata['feat'].to(device)

            fg_graph_node_feat = fg.ndata['feat'].to(device)
            fg_graph_edge_feat = fg.edata['feat'].to(device)
            
            output = model(g_feats, g, lg, fg, g_graph_node_feat,g_graph_edge_feat, lg_graph_node_feat, lg_graph_edge_feat,fg_graph_node_feat,fg_graph_edge_feat, device = device, resent = resent,pooling=pooling).cpu()

            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()
            for j in range(y.shape[1]):
                c_valid = np.ones_like(y[:, j], dtype=bool)
                c_label, c_pred = y[c_valid, j], output[c_valid, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)

                arr_label = torch.cat((arr_label,c_label),0)
                arr_pred = torch.cat((arr_pred,c_pred),0)
                    
            total_preds = torch.cat((total_preds, arr_pred), 0)
            total_labels = torch.cat((total_labels, arr_label), 0)
        
    AUC = roc_auc_score(total_labels.numpy().flatten(), total_preds.numpy().flatten())
    
    
    return AUC



def parse_arguments():
    parser = argparse.ArgumentParser(description="示例命令行工具")

    # 添加命令行参数
    parser.add_argument("--config", type=str, help="配置文件路径")

    args = parser.parse_args()
    args.config = './config/c_path.yaml'
    # 如果提供了配置文件路径，则加载配置文件
    if args.config:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)

        # 将配置文件中的参数添加到命令行参数中
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

    

    args = parse_arguments()
    for key, value in vars(args).items():
        if key != 'config':
            print(f"{key}: {value}")
    datafile = args.select_dataset
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    vali_ratio = args.vali_ratio
    test_ratio = args.test_ratio
    target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1}
    target_dim = target_map[datafile]

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
    

    
    creat_data(datafile, encoder_atom, encoder_bond,encode_two_path, encode_tree_path,batch_size, train_ratio, vali_ratio, test_ratio)

    model_select = args.model_select
    resent = args.resent
    pooling = args.pooling
    loss_sclect = args.loss_sclect


    # 加载 DataLoader 使用的数据集和其他必要信息
    state = torch.load('data/processed/'+datafile+'.pth')

    # 重新创建 CustomDataset 和 DataLoader
    loaded_train_dataset = CustomDataset(state['train_label'],state['train_graph_feat'], state['train_graph_list'])
    loaded_valid_dataset = CustomDataset(state['valid_label'],state['valid_graph_feat'], state['valid_graph_list'])
    loaded_test_dataset = CustomDataset(state['test_label'], state['test_graph_feat'],state['test_graph_list'])
    
   

    loaded_train_loader = DataLoader(loaded_train_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    if vali_ratio == 0.0:
        loaded_valid_loader = []
    else:
        loaded_valid_loader = DataLoader(loaded_valid_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    loaded_test_loader = DataLoader(loaded_test_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)


    print('dataset was loaded!')

    print("length of training set:",len(loaded_train_dataset))
    print("length of validation set:",len(loaded_valid_dataset))
    print("length of testing set:",len(loaded_test_dataset))
    
    iter = args.iter
    LR = args.LR
    num_blcok = args.num_blcok
    NUM_EPOCHS = args.NUM_EPOCHS
    #num_blcok = args.num_blcok
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    num_heads = args.head
    num_layers = args.layer_number
    tras_med = args.tras_med
    if datafile in ['tox21']:
        Random_seed = [2, 44, 46, 0, 42]
        seed = 42
    elif datafile in ['hiv']:
        Random_seed = [0, 1, 2, 43, 42]
        seed = 42

    else:
        Random_seed = [1,2,3,42,0]#0
        seed = 42
    set_seed(seed)
    print(seed)
    All_AUC = []
    for i in range(iter):
        # 设置种子
        #seed = Random_seed[i]
        
        AUC_list = []
        
        if model_select == 'pcnn':
            model = PCNN(in_feats=10, hidden_size = 32, out_feats=64, encode_dim=encode_dim, out_dim = target_dim,tras_med = tras_med,num_blcok=num_blcok,num_heads=num_heads,num_layers=num_layers)

        else:
            print('No found model!!!')
        
        #print(model)
        # 统计模型的总参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        train_loss_dic = {}
        vali_loss_dic = {}

        #model = modeling().to(device)
        model = model.to(device)
        if loss_sclect == 'l1':
            #loss_fn = nn.L1Loss()
            loss_fn = nn.L1Loss(reduction='sum')#sum，mean,none

        if loss_sclect == 'l2':
            loss_fn = nn.MSELoss(reduction='none')

        if loss_sclect == 'sml1':
            loss_fn = nn.SmoothL1Loss(reduction='sum')#mean,none,sum

        if loss_sclect == 'bce':
            loss_fn = nn.BCELoss(reduction='mean')#mean
        

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        best_auc = 0
        for epoch in range(NUM_EPOCHS):
            train_loss,vali_loss = train(model, device, loaded_train_loader, loaded_valid_loader, optimizer, epoch + 1)

            
            AUC = predicting(model, device, loaded_test_loader)
            
            
            if AUC > best_auc:
                best_auc = AUC
                logger.info(f'AUC: {best_auc:.5f}')
                formatted_number = "{:.5f}".format(best_auc)
                best_auc = float(formatted_number)
                AUC_list.append(best_auc)

                print(f"Epoch [{epoch+1}], Learning Rate: {scheduler.get_last_lr()}")

            if epoch % 10 == 0:
                #MAE_list.append(best_MAE)
                print("-------------------------------------------------------")
                print("epoch:",epoch)
                print('best_MAE:', best_auc)
            
            if epoch == NUM_EPOCHS-1:
                print(f"the best result up to {i+1}-loop is {best_auc:.4f}.")
                formatted_number = "{:.5f}".format(best_auc)
                All_AUC.append(best_auc)
    torch.save(model.state_dict(), 'model.pth')
    
    # 计算均值
    mean_value = statistics.mean(All_AUC)
    # 计算标准差
    std_dev = statistics.stdev(All_AUC)
    # 打印结果
    print(seed)
    print("均值:", mean_value)
    print("标准差:", std_dev)
