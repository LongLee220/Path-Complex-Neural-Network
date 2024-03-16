import dgl
import networkx as nx
import numpy as np


import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from jarvis.core.specie import chem_data, get_node_attributes
from torch_geometric.data import Data

def has_isolated_hydrogens(samiles):
    molecule = Chem.MolFromSmiles(samiles)
    # 获取分子中的原子
    atoms = molecule.GetAtoms()

    # 遍历原子
    for atom in atoms:
        # 如果原子是氢原子且没有邻居
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0:
            return True  # 存在孤立的氢原子

    return False  # 不存在孤立的氢原子


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

    angle_deg = np.degrees(angle_rad)

    return angle_deg


def Calculate_feature(dis_mat,all_atoms):
    Feat_mat = np.zeros((len(all_atoms),62),dtype=float)
    for a in range(len(dis_mat[0,:])):
        for b in range(len(dis_mat[:,0])):
            if dis_mat[a][b] == 0:
                continue
            if all_atoms[b] == 'C':
                if dis_mat[a][b] >= 10:
                    Feat_mat[a,18] += 1
                else:
                    Feat_mat[a,int((dis_mat[a,b]-1)*2)] += 1
            elif all_atoms[b] == 'H':
                if dis_mat[a][b] >= 10:
                    Feat_mat[a,37] += 1
                else:
                    Feat_mat[a,19+int((dis_mat[a,b]-1)*2)] += 1
            elif all_atoms[b] == 'O':
                if dis_mat[a][b] < 2.5:
                    Feat_mat[a,38] += 1
                elif dis_mat[a][b] < 5:
                    Feat_mat[a,39] += 1
                elif dis_mat[a][b] < 7.5:
                    Feat_mat[a,40] += 1
                else:
                    Feat_mat[a,41] += 1
            elif all_atoms[b] == 'N':
                if dis_mat[a][b] <=2.5:
                    Feat_mat[a,42] += 1
                elif dis_mat[a][b] < 5:
                    Feat_mat[a,43] += 1
                elif dis_mat[a][b] < 7.5:
                    Feat_mat[a,44] += 1
                else:
                    Feat_mat[a,45] += 1    
            elif all_atoms[b] == 'P':
                if dis_mat[a][b] < 5:
                    Feat_mat[a,46] += 1
                else:
                    Feat_mat[a,47] += 1
            elif all_atoms[b] == 'Cl'or all_atoms[a] == 'CL':
                if dis_mat[a][b] < 5:
                    Feat_mat[a,48] += 1
                else:
                    Feat_mat[a,49] += 1
            elif all_atoms[b] == 'F':
                if dis_mat[a][b] < 5:
                    Feat_mat[a,50] += 1
                else:
                    Feat_mat[a,51] += 1
            elif all_atoms[b] == 'Br':
                if dis_mat[a][b] < 5:
                    Feat_mat[a,52] += 1
                else:
                    Feat_mat[a,53] += 1
            elif all_atoms[b] == 'S':
                if dis_mat[a][b] < 5:
                    Feat_mat[a,54] += 1
                else:
                    Feat_mat[a,55] += 1
            elif all_atoms[b] == 'Si':
                if dis_mat[a][b] < 5:
                    Feat_mat[a,56] += 1
                else:
                    Feat_mat[a,57] += 1
            elif all_atoms[b] == 'I':
                if dis_mat[a][b] < 5:
                    Feat_mat[a,58] += 1
                else:
                    Feat_mat[a,59] += 1
            else:
                if dis_mat[a][b] < 5:
                    Feat_mat[a,60] += 1
                else:
                    Feat_mat[a,61] += 1
    return Feat_mat

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


def atom_to_graph(smiles):
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # 加氢
    smiles_with_hydrogens = Chem.MolToSmiles(mol)

    tmp = []
    for num in smiles_with_hydrogens:
        if num not in ['[',']','(',')']:
            tmp.append(num)

    sm = {}
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        sm[atom_index] =  atom.GetSymbol()



    Num_toms = len(tmp)
    sps_features = []
    coor = []

    # 添加构象信息
    AllChem.EmbedMolecule(mol, randomSeed=1)

    for ii, s in enumerate(mol.GetAtoms()):
        feat = list(get_node_attributes(s.GetSymbol(), atom_features='cgcnn'))
        sps_features.append(feat)
        
        # 获取原子坐标信息
        pos = mol.GetConformer().GetAtomPosition(ii)
        coor.append([pos.x, pos.y, pos.z])

    edge_features = []
    src_list, dst_list = [], []
    
    #edge_index = torch.LongTensor([[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4],
    #                           [1, 2, 4, 0, 2, 1, 0, 3, 2, 4, 3, 0]])
    x = torch.ones(5, 2)
    Distance_matrix = np.zeros((Num_toms,Num_toms),dtype=float)

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
        Distance_matrix[src][dst] = s_d_dis
        Distance_matrix[dst][src] = s_d_dis
        #edge_features.append([bond_type,s_d_dis])#[边类型，边长度]
        #edge_features.append([bond_type,s_d_dis])#[边类型，边长度]
        #edge_features.append([s_d_dis])#[边类型，边长度]
        #edge_features.append([s_d_dis])#[边类型，边长度]
        per_bond_feat = []
        per_bond_feat.extend(encode_bond(bond))
        edge_features.append(per_bond_feat)
        edge_features.append(per_bond_feat)

    feats = Calculate_feature(Distance_matrix,sm)

    node_feats = torch.tensor(feats,dtype=torch.float32)
    coor_tensor = torch.tensor(coor, dtype=torch.float32)
    edge_feats = torch.tensor(edge_features, dtype=torch.float32)
    edge_index = torch.LongTensor([src_list,dst_list])
    graph_attr = torch.tensor([0.9, 1.0], dtype=torch.float32)
    
    data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, graph_attr=graph_attr)


    # Number of atoms
    num_atoms = mol.GetNumAtoms()

    # Create a graph
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


def create_linear_graph(g, g_edges):
    # Create an undirected NetworkX graph from the DGL graph
    nx_g = g.to_networkx()

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
                    edge_feats.append([calculate_angle(A_coor, B_coor, C_coor)])
                    edge_feats.append([calculate_angle(A_coor, B_coor, C_coor)])

    G_nodes = torch.tensor(list(node_map.keys()))
    G = dgl.DGLGraph()
    G.add_edges(src, dst)
    G.ndata['feat'] = g.edata['feat'][G_nodes]
    G.edata['feat'] = torch.tensor(edge_feats, dtype=torch.float32)

    return G

def simplex_complex(Smile):
    #generate graph
    # 创建一个空的DGL图
    if has_isolated_hydrogens(Smile) == False:

        g, edge_index = atom_to_graph(Smile)
        G = create_linear_graph(g, edge_index)
        return g, G
    else:
        return print('存在孤立的氢原子')