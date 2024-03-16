import numpy as np
import dgl
import torch
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_3D_coordinates(mol):
    AllChem.Compute2DCoords(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)


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
    atom_type = [0] * 119
    atom_type[atom.GetAtomicNum() - 1] = 1
    
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
    valid_hybridization_types = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
    
    for i, hybridization_type in enumerate(valid_hybridization_types):
        if  hybridization_type in valid_hybridization_types:
            hybridization[i] = 1  

    return atom_type + aromaticity + formal_charge + chirality_tags + degree + num_hydrogens + hybridization



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



def encode_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # 加氢
    sts = Chem.FindMolChiralCenters(mol,includeUnassigned=True,force=False)

    atom_features = []
    coor = []
    # 添加构象信息
    AllChem.EmbedMolecule(mol, randomSeed=1)

    for ii, atom in enumerate(mol.GetAtoms()):
        per_atom_feat = []
        per_atom_feat.extend(encode_atom(atom))
        atom_features.append(per_atom_feat )
        # 获取原子坐标信息
        pos = mol.GetConformer().GetAtomPosition(ii)
        coor.append([pos.x, pos.y, pos.z])
    
    bond_features = []
    src_list, dst_list = [], []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondTypeAsDouble() # 存储键信息
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()

        src_list.append(src)
        src_list.append(dst)
        dst_list.append(dst)
        dst_list.append(src)
        per_bond_feat = []
        per_bond_feat.extend(encode_bond(bond))
        bond_features.append(per_bond_feat)
        bond_features.append(per_bond_feat)

    node_feats = torch.tensor(atom_features,dtype=torch.float32)
    coor_tensor = torch.tensor(coor, dtype=torch.float32)
    edge_feats = torch.tensor(bond_features, dtype=torch.float32)

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
                    edge_feats.append([calculate_angle(A_coor, B_coor, C_coor)])
                    edge_feats.append([calculate_angle(A_coor, B_coor, C_coor)])


    G = dgl.DGLGraph()
    G_nodes = torch.tensor(list(node_map.keys()))
    G.add_edges(src, dst)


    G.ndata['feat'] = g.edata['feat'][G_nodes]
    G.edata['feat'] = torch.tensor(edge_feats, dtype=torch.float32)

    return G


def mae_complex(Smile):
    #generate graph
    # 创建一个空的DGL图
    g, edge_index = encode_molecule(Smile)
    G = create_linear_graph(g, edge_index)

    return g, G