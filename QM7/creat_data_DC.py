import os
import numpy as np
import pandas as pd
import torch
import argparse



from splitters import ScaffoldSplitter
from cgcnn_path import path_complex
from encoder_simplex import simplex_complex
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from atom_graph import atom_to_graph
from rdkit import Chem


class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list):
        self.labels = label_list
        self.graphs = graph_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        graph_pair = self.graphs[index]

        # Assuming graph_pair is a list with two DGLGraphs
        graph1, graph2 = graph_pair

        return label, graph1, graph2


def has_isolated_hydrogens(samiles):
    # 获取分子中的原子
    molecule = Chem.MolFromSmiles(samiles)
    atoms = molecule.GetAtoms()
    # 遍历原子
    for atom in atoms:
        # 如果原子是氢原子且没有邻居
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0:
            return True  # 存在孤立的氢原子

    return False  # 不存在孤立的氢原子


def min_max_normalize(data):
    # 找到最小值和最大值
    min_val = min(data)
    max_val = max(data)

    # 对每个数据点应用Min-Max归一化公式
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]

    return normalized_data, min_val, max_val


def inverse_min_max_normalize(normalized_data, min_val, max_val):
    # 对每个归一化后的数据点应用逆操作
    original_data = [x * (max_val - min_val) + min_val for x in normalized_data]
    return original_data


def creat_data(datafile, encoder_meathed):
    
    datasets = datafile

    df = pd.read_csv('data/' + datasets + '.csv')#
    smiles_list, labels = df['smiles'], df['label']        
    #labels = labels.replace(0, -1)
    
    labels, min_val, max_val = min_max_normalize(labels)

    data_list = []
    for i in range(len(smiles_list)):
        smiles = smiles_list[i]

        if has_isolated_hydrogens(smiles) == False:
            if encoder_meathed == 'cgcnn':
                Graph_list = path_complex(smiles, i) 

            if encoder_meathed == 'self':
                Graph_list = simplex_complex(smiles)

            data_list.append([smiles, torch.tensor(labels.values[i]),Graph_list])


    #data_list = [['occr',albel,[c_size, features, edge_indexs],[g,liearn_g]],[],...,[]]

    print('Graph list was done!')

    splitter = ScaffoldSplitter().split(data_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    
    print('splitter was done!')
    

    
    train_label = []
    train_graph_list = []
    for tmp_train_graph in splitter[0]:
        
        train_label.append(tmp_train_graph[1])
        train_graph_list.append(tmp_train_graph[2])


    valid_label = []
    valid_graph_list = []
    for tmp_valid_graph in splitter[1]:
        valid_label.append(tmp_valid_graph[1])
        
        valid_graph_list.append(tmp_valid_graph[2])

    test_label = []
    test_graph_list = []
    for tmp_test_graph in splitter[2]:
        test_label.append(tmp_test_graph[1])
        test_graph_list.append(tmp_test_graph[2])

    batch_size = 256

    torch.save({
        'train_label': train_label,
        'train_graph_list': train_graph_list,
        'valid_label': valid_label,
        'valid_graph_list': valid_graph_list,
        'test_label': test_label,
        'test_graph_list': test_graph_list,
        'batch_size': batch_size,
        'shuffle': True,  # 保存时假设你在创建 DataLoader 时使用了 shuffle=True
        # 其他必要信息
    }, 'data/processed/'+ datafile +'.pth')

    print('preparing in pytorch format!')

    return min_val, max_val

if __name__ == "__main__":

    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description="示例命令行工具")

    # 添加命令行参数
    parser.add_argument("--select_dataset", type=str, help="select dataset")
    parser.add_argument("--encoder_meathed", type=str, help="encoder meathed")

    args = parser.parse_args()
    # 访问命令行参数的值
    datafile = args.select_dataset
    encoder_meathed = args.encoder_meathed
    creat_data(datafile, encoder_meathed)
    
