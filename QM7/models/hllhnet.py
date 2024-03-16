#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:41:15 2024

@author: longlee
"""

from multiprocessing import pool
import torch
import torch.nn as nn
import dgl.function as fn
import dgl
import numpy as np
import torch.nn.functional as F
import math
import dgl
import dgl.nn as dglnn
import numpy as np
import copy
import torch.nn.init as init
import sympy as sym



from models.basis_utils import bessel_basis, real_sph_harm, swish
from typing import Optional
from torch.nn import init
from torch_geometric.nn import MessagePassing, global_mean_pool
from sklearn.kernel_approximation import RBFSampler
from dgl.nn import GlobalAttentionPooling, SortPooling
from dgl.utils import expand_as_pair 
from dgl.nn.functional import edge_softmax
from dgl import DGLError


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")



def swish(x):
    """
    Swish activation function,
    from Ramachandran, Zopf, Le 2017. "Searching for Activation Functions"
    """
    return x * torch.sigmoid(x)




class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff
    """
    def __init__(self, exponent):
        super(Envelope, self).__init__()

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2
    
    def forward(self, x):
        # Envelope function divided by r
        x_p_0 = x.pow(self.p - 1)
        x_p_1 = x_p_0 * x
        x_p_2 = x_p_1 * x
        env_val = 1 / x + self.a * x_p_0 + self.b * x_p_1 + self.c * x_p_2
        return env_val



def GlorotOrthogonal(tensor, scale=2.0):
    if tensor is not None:
        nn.init.orthogonal_(tensor.data)
        scale /= (tensor.size(-2) + tensor.size(-1)) * tensor.var()
        tensor.data *= scale.sqrt()




def pool_subgraphs(out, batched_graph):
    # 将整图的输出按照子图数量拆分成子图的输出
    subgraphs = dgl.unbatch(batched_graph)

    # 对每个子图进行池化操作，这里使用了平均池化
    pooled_outputs = []
    ini = 0
    for subgraph in subgraphs:
        # 获取子图节点的数量
        num_nodes = subgraph.num_nodes()
        
        # 根据子图的节点数量对整图的输出进行切片
        start_idx = ini
        end_idx = start_idx + num_nodes
        sg_out = out[start_idx:end_idx]
        ini += num_nodes
        # 计算每个子图的平均池化表示
        #print(sg_out.shape)
        #pooled_out = F.avg_pool2d(sg_out, kernel_size=num_nodes)  # 使用节点数作为池化核的大小
        pooled_out = F.adaptive_avg_pool1d(sg_out.unsqueeze(0).permute(0, 2, 1), 1).squeeze()

        pooled_outputs.append(pooled_out)

    return torch.stack(pooled_outputs)





class RBFKernelMappingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, gamma):
        super(RBFKernelMappingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.random_state=42
        self.RBF = RBFSampler(gamma = self.gamma, n_components = self.output_dim,random_state=self.random_state)

    def forward(self, input_tensor):
        # 将输入张量移动到 CPU 上
        input_tensor_cpu = input_tensor.cpu()
        # 使用 RBF 核映射处理输入张量
        mapped_tensor_cpu = torch.tensor(self.RBF.fit_transform(input_tensor_cpu), dtype=torch.float32)
        # 将处理后的张量移动到 GPU 上
        mapped_tensor_gpu = mapped_tensor_cpu.to(input_tensor.device)
        return mapped_tensor_gpu




class RBFExpansion(nn.Module):
    """Expand input tensor with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        gamma: float = 0.5,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )
        self.gamma = gamma

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to input tensor."""
        return torch.exp(
            -self.gamma * (input_tensor - self.centers) ** 2
        )



class EGATlayer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads, bias=True):
        super(EGATlayer,self).__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_node = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats*num_heads, bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats*num_heads, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats*num_heads, bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()
        self
        
    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_node.weight, gain=gain)
        init.xavier_normal_(self.fc_ni.weight, gain=gain)
        init.xavier_normal_(self.fc_fij.weight, gain=gain)
        init.xavier_normal_(self.fc_nj.weight, gain=gain)
        init.xavier_normal_(self.attn, gain=gain)
        init.constant_(self.bias, 0)

    def forward(self, graph, nfeats, efeats, get_attention=False):
        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue.')
            f_ni = self.fc_ni(nfeats)
            f_nj = self.fc_nj(nfeats)
            f_fij = self.fc_fij(efeats)
            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            # add ni, nj factors
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            # add fij to node factor
            f_out = graph.edata.pop('f_tmp') + f_fij
            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = nn.functional.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
            # compute attention factor
            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)
            graph.edata['a'] = edge_softmax(graph, e)
            graph.ndata['h_out'] = self.fc_node(nfeats).view(-1, self._num_heads,
                                                             self._out_node_feats)
            # calc weighted sum
            graph.update_all(fn.u_mul_e('h_out', 'a', 'm'),
                             fn.sum('m', 'h_out'))

            h_out = graph.ndata['h_out'].view(-1, self._num_heads, self._out_node_feats)
            if get_attention:
                return h_out, f_out, graph.edata.pop('a')
            else:
                return h_out, f_out
            



class EGAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, out_dim,dropout=0.2, num_layers=2):
        super(EGAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.edge_feature_size = 10
        self.gcn_1_layers = nn.ModuleList()
        self.gcn_2_layers = nn.ModuleList()
        self.gcn_3_layers = nn.ModuleList()

        # RBF核函数的参数

        #第二层
        #self.edge_rbf_layer_2_path = RBFKernelMappingLayer(input_dim=1, output_dim=self.bond_feature_size, gamma=0.1)

        #第二层
        self.layer1 = EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=36,num_heads=8)

        self.gcn_1_layers.append(self.layer1)

        #第三层
        #self.edge_rbf_layer = RBFKernelMappingLayer(input_dim=self.bond_feature_size, output_dim=self.atom_feature_size, gamma=0.1)

        self.layer2= EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=8)

        self.gcn_2_layers.append(self.layer2 )
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=8))
        


        layers = [
            nn.Linear(self.hidden_size*2, out_feats),
            nn.Sigmoid(),
            nn.Linear(out_feats, self.out_dim),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout = nn.Sequential(*layers)
        self.Readout.apply(self.initialize_weights)
        self.apply(self.initialize_weights)


    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            GlorotOrthogonal(m.weight)  # 使用GlorotOrthogonal函数初始化权重
            if m.bias is not None:  # 检查偏置是否存在
                nn.init.zeros_(m.bias)  # 初始化偏置为0
            #nn.init.zeros_(m.bias)      # 初始化偏置为0

            
    def forward(self, lg, gg, device, resent, pooling):

        self.to(device)
        batch_outputs = []

        # 第二层GCN
        edge_feats = gg.edata['feat'].to(device)
        node_feats = gg.ndata['feat'].to(device)
        #edge_feats = (torch.tensor(self.edge_rbf_layer_2_path(edge_feats))).to(device)
        node_feats = node_feats.to(edge_feats.dtype).to(device)

        for layer in self.gcn_1_layers:
            node_feats, edge_feats = layer(gg.to(device), node_feats, edge_feats)

            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)
        node_feats = node_feats.to(device)
        
        y_2 = node_feats
        y_22 = edge_feats
        m, n = y_2.shape

        # 使用 torch.cat 生成 2m x n 张量 B
        B = torch.cat([y_2[i:i+1].repeat(2, 1) for i in range(m)], dim=0)
        y_2 = B.to(device)


        #第三层
        node_feats = lg.ndata['feat'].to(device)
        edge_feats = lg.edata['feat'].to(device)
        #edge_feats = (torch.tensor(self.edge_rbf_layer(edge_feats))).to(device)
        
        concatenated_tensor = torch.cat((y_2.unsqueeze(0), edge_feats.unsqueeze(0)), dim=0)
        edge_feats = torch.add(concatenated_tensor[0], concatenated_tensor[1])

        node_feats = node_feats.to(edge_feats.dtype).to(device)
        ini_node_feat = F.adaptive_avg_pool1d(node_feats.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        
        for layer in self.gcn_2_layers:
            node_feats,edge_feats = layer(lg.to(device), node_feats, edge_feats)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)

        y_11 = node_feats.to(device)    
        

        if pooling == 'avg':
            y_1 = pool_subgraphs(y_11, lg)
            y_2 = pool_subgraphs(y_22, gg)
            #y3 = pool_subgraphs(y_3, gg)

            #y_1 = F.adaptive_avg_pool1d(y_11.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        if resent == 'True':
            y = (y_1 + ini_node_feat).to(torch.float32)

        y = torch.cat((y_1, y_2), dim=1)

        batch_out = y.squeeze() 
        #batch_outputs.append(y)
        #batch_out = torch.stack(batch_outputs, dim=0).to(device)

        out = self.Readout(batch_out)
        return out
    





class HL_LH_block_1(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(HL_LH_block_1, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.gcn_1_layers = nn.ModuleList()
        self.gcn_2_layers = nn.ModuleList()
        self.gcn_3_layers = nn.ModuleList()
        self.gcn_4_layers = nn.ModuleList()
        
        
        #H_information to L_information 
        self.layer1 = EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=8)

        self.gcn_1_layers.append(self.layer1)

        self.linear = nn.Linear(self.bond_feature_size*2,self.bond_feature_size)

        self.layer2= EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=8)

        self.gcn_2_layers.append(self.layer2 )
        
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=8))

        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            GlorotOrthogonal(m.weight)  # 使用GlorotOrthogonal函数初始化权重
            if m.bias is not None:  # 检查偏置是否存在
                nn.init.zeros_(m.bias)  # 初始化偏置为0
            #nn.init.zeros_(m.bias) 

    def forward(self, lg, gg, device, node_feats,edge_feats,node_path,edge_path):
        

        # from H information to L information
        for layer in self.gcn_1_layers:
            node_path, edge_path = layer(gg.to(device), node_path, edge_path)
            node_path = node_path
            edge_path = edge_path
            node_path = torch.sum(node_path, dim=1)
            edge_path = torch.sum(edge_path, dim=1)
        node_path = node_path
        
        y_2 = node_path
        
        m, n = y_2.shape

        # 使用 torch.cat 生成 2m x n 张量 B
        B = torch.cat([y_2[i:i+1].repeat(2, 1) for i in range(m)], dim=0)
        edge_feats = B

        node_feats = node_feats.to(edge_feats.dtype)

        
        for layer in self.gcn_2_layers:
            node_feats,edge_feats = layer(lg.to(device), node_feats, edge_feats)
            node_feats = node_feats
            edge_feats = edge_feats
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)
            
           
        return node_feats, edge_feats
            

class HL_LH_block_2(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(HL_LH_block_2, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.gcn_3_layers = nn.ModuleList()
        self.gcn_4_layers = nn.ModuleList()
        
        
        #L_informatin to H_ingromation 
        self.layer3 = EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=8) 

        self.gcn_3_layers.append(self.layer3)
        for _ in range(num_layers - 1):
            self.gcn_3_layers.append(EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=8))
        
        self.linear_layer = nn.Linear(self.bond_feature_size*2,self.bond_feature_size)
        
        self.layer4= EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=8)

        self.gcn_4_layers.append(self.layer4)
  

        self.apply(self.initialize_weights)


    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            GlorotOrthogonal(m.weight)  # 使用GlorotOrthogonal函数初始化权重
            if m.bias is not None:  # 检查偏置是否存在
                nn.init.zeros_(m.bias)  # 初始化偏置为0
            #nn.init.zeros_(m.bias) 


    def forward(self, lg, gg, device, node_feats,edge_feats,node_path,edge_path):
        

        for layer in self.gcn_3_layers:
            node_feats, edge_feats= layer(lg.to(device), node_feats, edge_feats)
            node_feats = node_feats
            edge_feats = edge_feats
            node_feats= torch.sum(node_feats, dim=1)
            edge_feats= torch.sum(edge_feats, dim=1)
        node_feats = node_feats
        

        x_11 = torch.cat((edge_feats[0::2, :], edge_feats[1::2, :]), dim=1) 
        node_path = self.linear_layer(x_11)
        

        for layer in self.gcn_4_layers:
            node_path, edge_path= layer(gg.to(device), node_path, edge_path)
            node_path = node_path
            edge_path = edge_path
            node_path = torch.sum(node_path, dim=1)
            edge_path = torch.sum(edge_path, dim=1)

        return node_path, edge_path
    



class HL_LH_block(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(HL_LH_block, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.block_layers = nn.ModuleList()

        
        #H_information to L_information 
        self.layer1 = HL_LH_block_1(in_feats, hidden_size, out_feats, encode_dim)

        self.block_layers.append(self.layer1)

        
        #self.layer2= HL_LH_block_2(in_feats, hidden_size, out_feats, encode_dim)
        #self.block_layers.append(self.layer2)

        


    def forward(self, lg, gg, device, node_feats,edge_feats,node_path,edge_path):

        # from H information to L information
        for i in range(len(self.block_layers)):
            layer = self.block_layers[i]
            if i == 0:
                node_feats, edge_feats = layer(lg, gg, device, node_feats,edge_feats,node_path,edge_path)
            
            if i == 1:
                node_path, edge_path = layer(lg, gg, device, node_feats,edge_feats,node_path,edge_path)


        return node_feats, edge_feats, node_path, edge_path
    
    







class HL_LH_GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, out_dim,dropout=0.2, num_blcok=1,activation=swish):
        super(HL_LH_GAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_blcok = num_blcok
        self.out_dim = out_dim
        self.activation = activation
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.blockconv = nn.ModuleList()

        '''
        self.rbf_layer = BesselBasisLayer(num_radial=self.hidden_size,
                                          cutoff=5,
                                          envelope_exponent=5).to(device) 
        
        self.sbf_layer = SphericalBasisLayer(num_spherical=7,
                                             num_radial=self.hidden_size,
                                             cutoff=5,
                                             envelope_exponent=5).to(device) 
        
        self.emb_block_1 = EmbeddingBlock_LG(emb_size=self.hidden_size,
                                        num_radial=self.hidden_size,
                                        bessel_funcs=self.sbf_layer.get_bessel_funcs(),
                                        cutoff=5,
                                        envelope_exponent=5,
                                        activation=activation).to(device) 
        
        self.emb_block_2 = EmbeddingBlock_GG(emb_size=self.hidden_size,
                                        num_radial=self.hidden_size,
                                        bessel_funcs=self.sbf_layer.get_bessel_funcs(),
                                        cutoff=5,
                                        envelope_exponent=5,
                                        activation=activation).to(device) 

        '''
        self.edge_rbf_path = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.in_feats, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        ).to(device) 

        self.node_rbf_path = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.bond_feature_size, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        ).to(device) 


        self.node_rbf = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.atom_feature_size, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        ).to(device) 

        self.edge_rbf = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.bond_feature_size, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        ).to(device) 


        for i in range(self.num_blcok):
            self.blockconv.append(HL_LH_block(in_feats=64, hidden_size=64, out_feats=64, encode_dim=[64,64]))
        
        self.Linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        
        self.leaky_relu = nn.LeakyReLU()

        self.norm = nn.BatchNorm1d(self.hidden_size*2)

        layers = [
            nn.Linear(self.hidden_size*2, out_feats),
            self.leaky_relu,
            #self.activation,
            nn.Linear(out_feats, self.out_dim),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout = nn.Sequential(*layers)
        self.Readout.apply(self.initialize_weights)
        self.apply(self.initialize_weights)
        #self.reset_params()
    
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            GlorotOrthogonal(m.weight)  # 使用GlorotOrthogonal函数初始化权重
            if m.bias is not None:  # 检查偏置是否存在
                nn.init.zeros_(m.bias)  # 初始化偏置为0
            #nn.init.zeros_(m.bias) 

    '''
    def reset_params(self):
        GlorotOrthogonal(self.edge_rbf_path.weight)
        GlorotOrthogonal(self.node_rbf_path.weight)
        GlorotOrthogonal(self.edge_rbf.weight)
        GlorotOrthogonal(self.node_rbf.weight)

        for layer in self.blockconv:
            GlorotOrthogonal(layer.weight)
        self.output_init(nn.init.zeros_)
    '''

        
    def forward(self, lg, gg, device, resent, pooling):
        
        
        batch_outputs = []
        '''
        G_graph = gg.to(device) 
        G_graph = self.rbf_layer(G_graph)
        G_graph = self.emb_block_2(G_graph)
        node_path = G_graph.ndata['feat']
        edge_path = G_graph.edata['feat']

        L_graph = lg.to(device) 
        L_graph = self.rbf_layer(L_graph)
        L_graph = self.emb_block_1(L_graph)
        node_feats = L_graph.ndata['feat']
        edge_feats = L_graph.edata['feat']
        '''
        
        edge_path = gg.edata['feat'].to(device)
        node_path = gg.ndata['feat'].to(device)
        
        edge_path = self.edge_rbf_path(edge_path)

        node_path = self.node_rbf_path(node_path)

        node_path = node_path.to(edge_path.dtype).to(device)
        edge_path = edge_path.to(device)

        ini_y3 = self.activation(edge_path)
        ini_y2 = self.activation(node_path)

        
        node_feats = lg.ndata['feat'].to(device)
        edge_feats = lg.edata['feat'].to(device)

        edge_feats = self.edge_rbf(edge_feats)

        node_feats = self.node_rbf(node_feats)
        node_feats = node_feats.to(edge_feats.dtype).to(device)

    
        ini_y1 = self.activation(node_feats)
        
        # 在需要生成全零张量的地方
        
        out_1 = torch.zeros_like(node_feats)
        out_2 = torch.zeros_like(node_path)
        out_3 = torch.zeros_like(edge_path)
        for i in range(self.num_blcok):
            bolck = self.blockconv[i]
            node_feats,edge_feats,node_path,edge_path = bolck(lg,gg, device, node_feats,edge_feats,node_path,edge_path)
            

            edge_embedd = torch.cat((edge_feats[0::2, :], edge_feats[1::2, :]), dim=1)
            edge_embedd = self.Linear(edge_embedd)
            node_feats = node_feats
            edge_path = edge_path


            out_1 = out_1 + node_feats
            out_2 = out_2 + edge_embedd
            out_3 = out_3 + edge_path
            '''
            if i == 0:
                tensor_1 = out_1
            if i == self.num_blcok -1:
                tensor_2 = node_feats
            '''


        if resent == 'True':
            y_1 = (out_1 + ini_y1).to(torch.float32).to(device)
            y_2 = (out_2 + ini_y2).to(torch.float32).to(device)
            y_3 = (out_3 + ini_y3).to(torch.float32).to(device)
            
            #y_1 = torch.cat([tensor_1, tensor_2, ini_y1], dim=1)
        else:
            y_1 = (out_1).to(torch.float32).to(device)
            y_2 = (out_2).to(torch.float32)
            y_3 = (out_3).to(torch.float32)

            #y_1 = torch.cat([tensor_1, tensor_2], dim=1)
        
        #print(y_1.shape)
        if pooling == 'avg':
            y1 = pool_subgraphs(y_1, lg)
            y3 = pool_subgraphs(y_3, gg)
            
            y = torch.cat((y1, y3), dim=1)

        batch_outputs.append(y)
        batch_out = y.squeeze() 
        #print(batch_out.shape)
        out = self.norm(batch_out)
        out = self.Readout(out)
        return out