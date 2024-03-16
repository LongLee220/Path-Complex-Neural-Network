#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 14:59:54 2024

@author: longlee
"""

import dgl
import torch
import numpy as np
import networkx as nx


from rdkit import Chem
from rdkit.Chem import AllChem
from jarvis.core.specie import chem_data, get_node_attributes



def calculate_dis(A,B):
    AB = B - A
    dis = np.linalg.norm(AB)
    return dis


def calculate_angle(A, B, C):
    # vector(AB), vector(BC),# angle = arccos(AB*BC/|AB||BC|)
    AB = B - A
    BC = C - B

    dot_product = np.dot(AB, BC)
    norm_AB = np.linalg.norm(AB)
    norm_BC = np.linalg.norm(BC)
    if norm_AB * norm_BC != 0:
        cos_theta = dot_product / (norm_AB * norm_BC)
    else:
        # 处理分母为零的情况，例如给一个默认值或进行其他操作
        cos_theta = 0.0  # 你可以根据具体情况调整这里的值
    #cos_theta = dot_product / (norm_AB * norm_BC)
    # 假设 cos_theta 是你的输入
    if -1 <= cos_theta <= 1:
        angle_rad = np.arccos(cos_theta)
    else:
    # 在这里处理无效的输入，例如给出一个默认值或者引发异常
        angle_rad = 0  # 或者 raise ValueError("Invalid cos_theta value")

    #angle_rad = np.arccos(cos_theta)

    #angle_deg = np.degrees(angle_rad)

    return angle_rad

def calculate_triangle_properties(point1, point2, point3):
    point1, point2, point3 = point1.cpu().numpy(), point2.cpu().numpy(), point3.cpu().numpy()
    pps = []
    # 计算质心坐标
    centroid = np.mean([point1, point2, point3], axis=0)
    #pps.append(centroid)
    # 计算坐标到质心的距离
    distances = [np.linalg.norm(point - centroid) for point in [point1, point2, point3]]
    pps.extend(distances)
    # 计算三角形的边长
    a = np.linalg.norm(point2 - point1)
    b = np.linalg.norm(point3 - point2)
    c = np.linalg.norm(point1 - point3)
    pps.extend([a,b,c])

    # 计算三角形的角度（弧度）
    A = calculate_angle(point1, point2, point3)
    B = calculate_angle(point2, point1, point3)
    C = calculate_angle(point3, point2, point1)
    pps.extend([A,B,C])

    
    if a > 0 and b > 0 and c > 0 and (a + b > c) and (a + c > b) and (b + c > a):
        s = 0.5 * (a + b + c)  # 半周长
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        pps.append(area)
    else:
        pps.append(0)
    '''
    if a > 0:
        if b > 0: 
            if c > 0:

                A = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                B = np.arccos((a**2 + c**2 - b**2) / (2 * a * c))
                C = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
                pps.extend([A,B,C])
                # 计算三角形的面积
                s = 0.5 * (a + b + c)  # 半周长
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                pps.append(area)

            else:
                if a > 0:
                    if b >0:
                        C = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
    '''
    return(pps)

def calculate_dihedral_angle(A, B, C, D):
    # vector(AB), vector(BC),# angle = arccos(AB*BC/|AB||BC|)
    BA = A - B
    BC = C - B
    CD = D - C
    N1 = np.cross(BA, BC)
    N2 = np.cross(BC, CD)
    norm_N1 = np.linalg.norm(N1)
    norm_N2 = np.linalg.norm(N2)
    dot_product = np.dot(N1, N2)

    cos_theta = dot_product / (norm_N1 * norm_N2)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg



def encode_chirality(atom):
    chirality_tags = [0] * 4  # Assuming 4 possible chirality tags
    
    if atom.HasProp("_CIPCode"):
        chirality = atom.GetProp("_CIPCode")
        if chirality == "R":
            chirality_tags[0] = 1
        elif chirality == "S":
            chirality_tags[1] = 1
        elif chirality == "E":
            chirality_tags[2] = 1
        elif chirality == "Z":
            chirality_tags[3] = 1
    
    return chirality_tags

def encode_atom(atom):
    #atom_type = [0] * 119
    #atom_type[atom.GetAtomicNum() - 1] = 1
    
    aromaticity = [0, 0]
    aromaticity[int(atom.GetIsAromatic())] = 1

    formal_charge = [0] * 16
    formal_charge[atom.GetFormalCharge() + 8] = 1  # Assuming formal charges range from -8 to +8

    chirality_tags = encode_chirality(atom)

    degree = [0] * 11  # Assuming degrees up to 10
    degree[atom.GetDegree()] = 1

    num_hydrogens = [0] * 9  # Assuming up to 8 hydrogens
    num_hydrogens[atom.GetTotalNumHs()] = 1

    hybridization = [0] * 5  # Assuming 5 possible hybridization types
    hybridization_type = atom.GetHybridization()
    valid_hybridization_types = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
    if  hybridization_type > 5:
        print(hybridization_type)
    for i, hybridization_type in enumerate(valid_hybridization_types):
        if  hybridization_type in valid_hybridization_types:
            hybridization[i] = 1  
    #atom_type +
    return aromaticity + formal_charge + chirality_tags + degree + num_hydrogens + hybridization



def bond_length_approximation(bond_type):
    bond_length_dict = {"SINGLE": 1.0, "DOUBLE": 1.4, "TRIPLE": 1.8, "AROMATIC": 1.5}
    return bond_length_dict.get(bond_type, 1.0)

def encode_bond(bond):
    #7+4+2+1 = 14
    bond_dir = [0] * 7
    bond_dir[bond.GetBondDir()] = 1
    
    bond_type = [0] * 4
    bond_type[int(bond.GetBondTypeAsDouble()) - 1] = 1
    
    bond_length = bond_length_approximation(bond.GetBondType())
    
    in_ring = [0, 0]
    in_ring[int(bond.IsInRing())] = 1
 
    return bond_dir + bond_type + [bond_length] + in_ring



def get_bond_formal_charge(bond):
    # 获取连接到键两端的原子
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()

    # 获取原子的形式电荷并相加
    bond_formal_charge = atom1.GetFormalCharge() + atom2.GetFormalCharge()
    return bond_formal_charge

def bond_length_approximation(bond_type):
    bond_length_dict = {"SINGLE": 1.0, "DOUBLE": 1.4, "TRIPLE": 1.8, "AROMATIC": 1.5}
    return bond_length_dict.get(bond_type, 0.5)


def is_polar_bond(bond):
    # 获取连接到键两端的原子
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()

    # 获取原子的电负性，这里使用的是 Pauling 电负性表
    electronegativity_table = {1: 2.20, 2: 0.98,3: 1.57, 4: 2.04, 5: 2.55, 6: 3.04, 7: 3.44, 8: 3.98, 9: 0.93, 10: 1.31,11: 1.61, 12: 1.90, 13: 2.19, 14: 2.58, 15: 3.16,16: 3.98, 17: 0.82, 18: 1.00, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54,23: 1.63, 24: 1.66,25: 1.55,26: 1.83, 27: 1.91, 28: 1.90, 29: 1.65, 30: 1.81, 31: 2.01, 32: 2.18, 33: 2.55, 34: 2.96, 35: 3.00, 36: 0.79, 37: 0.89, 38: 1.10, 39: 1.12,  40: 1.13, 41: 1.14, 42: 1.13, 43: 1.17, 44: 1.20, 45: 1.20, 46: 1.10, 47: 1.22, 48: 1.23, 49: 1.24, 50: 1.25, 51: 1.10, 52: 1.27, 53: 1.3, 54: 1.5, 55: 2.36, 56: 1.9, 57: 2.2, 58: 2.20,59: 2.28, 60: 2.54, 61: 2.00, 62: 1.62, 63: 2.33, 64: 2.02, 65: 1.30, 66: 1.50, 67: 1.38,68: 1.36, 69: 1.28, 70: 1.30,71: 1.30, 72: 1.30, 73: 1.30,74: 1.30, 75: 1.30, 76: 1.30, 77: 1.30,78: 1.30, 79: 1.30,80: 1.30, 81: 1.30, 82: 1.30, 83: 1.30, 84: 1.30, 85: 1.30, 86: 1.30, 87: 1.30, 88: 1.30, 89: 1.30, 90: 1.30, 91: 1.30, 92: 1.30, 93: 1.30, 94: 1.30,95: 1.30, 96: 1.30,97: 1.30,98: 1.30, 99: 1.30, 100: 1.30, 101: 1.30, 102: 1.30, 103: 1.30, 104: 1.30, 105: 1.30, 106: 1.30, 107: 1.30, 108: 1.30, 109: 1.30, 110: 1.30, 111: 1.30, 112: 1.30, 113: 1.30,114: 1.30, 115: 1.30,116: 1.30, 117: 1.30, 118: 1.30 }

    electronegativity_atom1 = electronegativity_table.get(atom1.GetAtomicNum(), 0.0)
    electronegativity_atom2 = electronegativity_table.get(atom2.GetAtomicNum(), 0.0)

    # 判断键是否为极性键
    is_polar = (electronegativity_atom1 * electronegativity_atom2)

    return is_polar



def is_rotatable_bond(bond):
    # 判断键是否可旋转的简单示例
    if bond.GetBondType() in [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.AROMATIC]:
        return True
    else:
        return False
'''
def encode_bond(bond):
    bond_encoding = {
        'topological_distance': bond.GetBeginAtom().GetDegree() + bond.GetEndAtom().GetDegree() - 2,
        'topological_radius': max(bond.GetBeginAtom().GetDegree(), bond.GetEndAtom().GetDegree()),
        'is_conjugated': bond.GetIsConjugated(),
        'is_rotatable': is_rotatable_bond(bond),
        'is_linear': bond.GetIsAromatic(),
        'bond_formal_charge': get_bond_formal_charge(bond),
        'is_polar': is_polar_bond(bond),
        'begin_atom_hybridization': bond.GetBeginAtom().GetHybridization(),
        'end_atom_hybridization': bond.GetBeginAtom().GetHybridization(),
        'begin_atom_partial_charge': bond.GetBeginAtom().GetProp('_GasteigerCharge') if bond.GetBeginAtom().HasProp('_GasteigerCharge') else 0,
        'end_atom_partial_charge': bond.GetEndAtom().GetProp('_GasteigerCharge') if bond.GetEndAtom().HasProp('_GasteigerCharge') else 0,
        'begin_atom_is_aromatic': bond.GetBeginAtom().GetIsAromatic(),
        'end_atom_is_aromatic': bond.GetEndAtom().GetIsAromatic(),
        'dipole_moment': bond.GetDipoleMoment().Length() if bond.HasProp('_CIPCode') else 0,
        'neighbor_count': len(bond.GetBeginAtom().GetNeighbors()) + len(bond.GetEndAtom().GetNeighbors()),
        'bond_types': bond_length_approximation(bond.GetBondType()),
        'begin_atom_degree':bond.GetBeginAtom().GetDegree(),
        'end_atom_degree':bond.GetEndAtom().GetDegree(),
        'begin_atom_types': bond.GetBeginAtom().GetAtomicNum(),
        'end_atom_types': bond.GetEndAtom().GetAtomicNum(),
        'neighbor_bond_orders': bond.GetBondTypeAsDouble(),
        'neighbor_ring_sizes': bond.IsInRing()
        # 添加其他键的性质
    }
    return bond_encoding
'''
def convert_to_float(dictionary):
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            if isinstance(value, (int, bool)):  # 将整数和布尔值转换为浮点数
                dictionary[key] = float(value)
            elif isinstance(value, list):  # 对列表中的每个元素递归进行转换
                dictionary[key] = [convert_to_float(item) for item in value]
    return dictionary


def encode_molecule(bond):
    bond_encoding = encode_bond(bond)
    # 转换后的结果列表
    # 转换后的结果列表
    converted_results = convert_to_float(bond_encoding)
    return list(converted_results.values())

def atom_to_graph(smiles,encoder):
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # 加氢
    sps_features = []
    coor = []

    # 添加构象信息
    #AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)

    for ii, s in enumerate(mol.GetAtoms()):

        '''
        feat = list(get_node_attributes(s.GetSymbol(), atom_features=encoder))
        sps_features.append(feat)
        '''
        per_atom_feat = []
        for encoder_w in encoder:
            if encoder_w != 'gme':
                feat = list(get_node_attributes(s.GetSymbol(), atom_features=encoder_w))
                per_atom_feat.extend(feat)
            else:
                per_atom_feat.extend(encode_atom(s))
                
        sps_features.append(per_atom_feat )
        
        
        # 获取原子坐标信息
        pos = mol.GetConformer().GetAtomPosition(ii)
        coor.append([pos.x, pos.y, pos.z])

    edge_features = []
    src_list, dst_list = [], []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondTypeAsDouble() # 存储键信息
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()

        src_list.append(src)
        src_list.append(dst)
        dst_list.append(dst)
        dst_list.append(src)

        src_coor = np.array(coor[src])
        dst_coor = np.array(coor[dst])
        s_d_dis = calculate_dis(src_coor,dst_coor)
        #edge_features.append([bond_type,s_d_dis])#[边类型，边长度]
        #edge_features.append([bond_type,s_d_dis])
        per_bond_feat = []
        per_bond_feat.extend(encode_bond(bond))
        edge_features.append(per_bond_feat)
        edge_features.append(per_bond_feat)

    node_feats = torch.tensor(sps_features,dtype=torch.float32)
    coor_tensor = torch.tensor(coor, dtype=torch.float32)
    edge_feats = torch.tensor(edge_features, dtype=torch.float32)

    # Number of atoms
    num_atoms = mol.GetNumAtoms()

    # Create a graph. undirected_graph
    g = dgl.DGLGraph()
    g.add_nodes(num_atoms)
    g.add_edges(src_list, dst_list)

    # Assign features to nodes and edges
    g.ndata['feat'] = node_feats
    g.ndata['coor'] = coor_tensor  # 添加原子坐标信息
    g.edata['feat'] = edge_feats

    edge_src, edge_dst = g.edges()
    edge_list = list(zip(edge_src.tolist(), edge_dst.tolist()))

    return g, edge_list


def normalize_columns_01(input_tensor):
    # Assuming input_tensor is a 2D tensor (matrix)
    min_vals, _ = input_tensor.min(dim=0)
    max_vals, _ = input_tensor.max(dim=0)

    # Identify columns where max and min are not equal
    non_zero_mask = max_vals != min_vals

    # Avoid division by zero
    normalized_tensor = input_tensor.clone()
    normalized_tensor[:, non_zero_mask] = (input_tensor[:, non_zero_mask] - min_vals[non_zero_mask]) / (max_vals[non_zero_mask] - min_vals[non_zero_mask] + 1e-10)

    return normalized_tensor


def create_linear_graph(g, g_edges):
    # Create an undirected NetworkX graph from the DGL grap
    node_feats = g.edata['feat']
    coor_feats = g.ndata['feat']
    # Iterate over edges in the original graph
    num_edges = len(g_edges)


    # Create a new graph G for edges
    nx_G = nx.Graph()
    
    edge_map = {}
    edge_feats = []
    src, dst = [], []
    edge_check = []
    node_map = {}
    tmp = 0
    for i in range(num_edges):
        edge_i = g_edges[i]
        if set(edge_i) not in edge_check:
            edge_check.append(set(edge_i))
            edge_map[edge_i] = i
            node_map[i] = tmp
            tmp += 1

        for j in range(i+1, num_edges):
            edge_j = g_edges[j]
            if set(edge_j) not in edge_check:
                edge_check.append(set(edge_j))
                edge_map[edge_j] = j
                node_map[j] = tmp
                tmp += 1

            common_neighbors = set(edge_i).intersection(set(edge_j))

            if len(common_neighbors) == 1 :
                if edge_i in edge_map.keys():

                    edge_i_map = edge_map[edge_i]#i
                else:
                    edge_i_map = edge_map[edge_i[1],edge_i[0]]#i

                if edge_j in edge_map.keys():

                    edge_j_map = edge_map[edge_j]#j
                else:
                    edge_j_map = edge_map[edge_j[1],edge_j[0]]#j


                node_i = node_map[edge_i_map]
                node_j = node_map[edge_j_map]

                if (node_i, node_j) not in nx_G.edges():


                    nx_G.add_edge(node_i, node_j)

                    src.append(node_i)
                    dst.append(node_j)
                    src.append(node_j)
                    dst.append(node_i)

                  
                    A,B,C = list(set(edge_i).union(set(edge_j)))
                    A_coor = g.ndata['coor'][A]
                    B_coor = g.ndata['coor'][B]
                    C_coor = g.ndata['coor'][C]
                    
                    pps = calculate_triangle_properties(A_coor, B_coor, C_coor)
                    edge_feats.append(pps)
                    edge_feats.append(pps)
                    '''
                    edge_feats.append([calculate_angle(A_coor, B_coor, C_coor)])
                    edge_feats.append([calculate_angle(A_coor, B_coor, C_coor)])
                    '''

    G = dgl.DGLGraph()
    G_nodes = torch.tensor(list(node_map.keys()))
    G.add_edges(src, dst)

    if G.number_of_nodes() != len(node_map):
        return False
    
    edge_feats = torch.tensor(edge_feats, dtype=torch.float32)

    G.ndata['feat'] = g.edata['feat'][G_nodes]
    #G.edata['feat'] = normalize_columns_01(edge_feats) 
    G.edata['feat'] = edge_feats

    return G

def dgl_graph(nodes_feat_dic,edges_feat_dic):
    
    # 提取节点和边的索引以及特征
    nodes = list(nodes_feat_dic.keys())
    edges, edge_features_list = zip(*edges_feat_dic.items())
    edge_src, edge_dst = zip(*edges)
    node_feats = torch.stack([nodes_feat_dic[node] for node in nodes])
    edge_feats = torch.stack(edge_features_list)


    # 构建dgl.graph
    g = dgl.graph(([],[]))
    g.add_nodes(len(nodes))
    g = dgl.graph((edge_src, edge_dst))
    if len(node_feats) != g.number_of_nodes():
        print(len(node_feats), g.number_of_nodes())
        print(node_feats)
        print(edge_feats)
        
    assert len(node_feats) == g.number_of_nodes()
    g.ndata['feat'] = node_feats
    g.edata['feat'] = edge_feats
    g.ndata['coor'] = torch.zeros((g.number_of_nodes(), 3))  # 初始化3表示坐标的维度
    return g



#def path_complex(Smile,Coor,drug_feat,edges_feat,edge_index):
def path_complex(Smile, encoder):
    #generate graph
    # 创建一个空的DGL图
    g, edge_index = atom_to_graph(Smile,encoder)
    G = create_linear_graph(g, edge_index)

    return g, G
