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




class RBFKernelMappingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, gamma):
        super(RBFKernelMappingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.random_state=42
        self.RBF = RBFSampler(gamma = self.gamma, n_components = self.output_dim,random_state=self.random_state)

    def forward(self, input_tensor):
        input_tensor_cpu = input_tensor.cpu()
        mapped_tensor = torch.tensor(self.RBF.fit_transform(input_tensor_cpu), dtype=torch.float32)
        mapped_tensor = mapped_tensor.to(device)
        return mapped_tensor






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



'''


class RBFKernelMappingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, gamma):
        super(RBFKernelMappingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.random_state = 42
        # 初始化 RBFSampler，但不拟合
        self.RBF = RBFSampler(gamma=self.gamma, n_components=self.output_dim, random_state=self.random_state)

    def forward(self, input_tensor):
        # 在第一次前向传播时拟合
        if not hasattr(self, 'centers_'):
            # 使用均匀分布的数据进行拟合
            random_data = np.random.randn(1, self.input_dim)

            #random_data = np.random.uniform(size=(1, self.input_dim))
            # 在拟合后保存核心点
            self.centers_ = self.RBF.fit(random_data).random_offset_

        # 在每次前向传播时进行变换
        mapped_tensor = torch.tensor(self.RBF.transform(input_tensor), dtype=torch.float32)
        return mapped_tensor


'''

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
        self.to(device)
        
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
        self.layer1 = EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2)

        self.gcn_1_layers.append(self.layer1)

        #第三层
        #self.edge_rbf_layer = RBFKernelMappingLayer(input_dim=self.bond_feature_size, output_dim=self.atom_feature_size, gamma=0.1)

        self.layer2= EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2)

        self.gcn_2_layers.append(self.layer2 )
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2))
        
        self.gate_nn = nn.Linear(self.atom_feature_size, 1)
        self.global_attention_pooling = GlobalAttentionPooling(self.gate_nn)
        self.sort_pooling = SortPooling(k=1)

        self.layer3 = EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2)

        self.gcn_3_layers.append(self.layer1)


        layers = [
            nn.Linear(self.atom_feature_size, out_feats),
            nn.Sigmoid(),
            nn.Linear(out_feats, self.out_dim),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout = nn.Sequential(*layers)

    def forward(self, lg, gg,fg, device, resent, pooling):

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
        
        m, n = y_2.shape

        # 使用 torch.cat 生成 2m x n 张量 B
        #B = torch.cat([y_2[i:i+1].repeat(2, 1) for i in range(m)], dim=0)
        #y_2 = B.to(device)


        #第三层
        node_feats = lg.ndata['feat'].to(device)
        edge_feats = lg.edata['feat'].to(device)
        #edge_feats = (torch.tensor(self.edge_rbf_layer(edge_feats))).to(device)
        
        #concatenated_tensor = torch.cat((y_2.unsqueeze(0), edge_feats.unsqueeze(0)), dim=0)
        edge_feats = torch.add(y_2.unsqueeze(0), edge_feats.unsqueeze(0))

        node_feats = node_feats.to(edge_feats.dtype).to(device)
        ini_node_feat = F.adaptive_avg_pool1d(node_feats.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        
        for layer in self.gcn_2_layers:
            node_feats,edge_feats = layer(lg.to(device), node_feats, edge_feats)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)

        y_11 = node_feats.to(device)    
        if pooling == 'glob':
            y_1 = self.global_attention_pooling(lg.to(device),y_11).squeeze(0)
        if pooling == 'sort':
            y_1 = self.sort_pooling(lg.to(device),y_11).squeeze(0)
            y_1 = y_1.squeeze(0)
        if pooling == 'avg':
            y_1 = pool_subgraphs(y_11, lg)
            #y3 = pool_subgraphs(y_3, gg)
            #y_1 = F.adaptive_avg_pool1d(y_11.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        if resent == 'True':
            y = (y_1 + ini_node_feat).to(torch.float32)
            
        else:
            y = (y_1 ).to(torch.float32)
        batch_outputs.append(y)
        batch_out = torch.stack(batch_outputs, dim=0).to(device)

        out = self.Readout(y)

        return out
    
    

class HL_blcok(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(HL_blcok, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.edge_feature_size = 10
        self.gcn_1_layers = nn.ModuleList()
        self.gcn_2_layers = nn.ModuleList()
        
        
        #第二层
        self.layer1 = EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2)

        self.gcn_1_layers.append(self.layer1)

        self.linear_layer = nn.Linear(self.bond_feature_size*2,self.bond_feature_size)
        
        self.layer2= EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2)

        self.gcn_2_layers.append(self.layer2 )
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2))          
            
    def forward(self, g_list, device, node_feats,edge_feats,node_path,edge_path):
         
         for layer in self.gcn_1_layers:
             node_path, edge_path = layer(g_list[1].to(device), node_path, edge_path)
             node_feats = node_path.to(device)
             edge_path = edge_path.to(device)
             node_path = torch.sum(node_path, dim=1)
             edge_path = torch.sum(edge_path, dim=1)
         node_path = node_path.to(device)
         
         y_2 = node_feats
         
         m, n = y_2.shape

         # 使用 torch.cat 生成 2m x n 张量 B
         B = torch.cat([y_2[i:i+1].repeat(2, 1) for i in range(m)], dim=0)
         y_2 = B.to(device)

         concatenated_tensor = torch.cat((y_2, edge_feats), dim=0)

         edge_feats = self.linear_layer(concatenated_tensor)

         node_feats = node_feats.to(edge_feats.dtype).to(device)

         for layer in self.gcn_2_layers:
             node_feats,edge_feats = layer(g_list[0].to(device), node_feats, edge_feats)
             node_feats = node_feats.to(device)
             edge_feats = edge_feats.to(device)
             node_feats = torch.sum(node_feats, dim=1)
             edge_feats = torch.sum(edge_feats, dim=1)
             
             
         return node_feats, edge_feats, node_path, edge_path
    
    
    

class HL_GAT(nn.Module):
    '''
    no RBF and information from high to low
    '''
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_blcok=1):
        super(HL_GAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_blcok = num_blcok
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.edge_feature_size = 10
        self.blockconv = nn.ModuleList()

        
        '''
        self.edge_rbf_path = RBFKernelMappingLayer(input_dim=self.in_feats, output_dim=self.hidden_size, gamma=0.5)
        self.Linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.node_rbf_path = RBFKernelMappingLayer(input_dim=self.bond_feature_size, output_dim=self.hidden_size, gamma=0.5)
        self.Linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        #第三层
        self.node_rbf = RBFKernelMappingLayer(input_dim=self.atom_feature_size, output_dim=self.hidden_size, gamma=0.5)
        self.Linear3 = nn.Linear(self.hidden_size, self.hidden_size)

        self.edge_rbf = RBFKernelMappingLayer(input_dim=self.bond_feature_size, output_dim=self.hidden_size, gamma=0.5)
        self.Linear4 = nn.Linear(self.hidden_size, self.hidden_size)
        '''
        
        for i in range(self.num_blcok):
            self.blockconv.append(HL_blcok(in_feats, hidden_size, out_feats, encode_dim))
        
        self.relu = nn.ReLU()

        layers = [
            nn.Linear(self.atom_feature_size, out_feats),
            nn.Sigmoid(),
            #NormalizeLayer(),
            nn.Linear(out_feats, 1),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout = nn.Sequential(*layers)




    def forward(self, g_list, device, resent, pooling):

        edge_feats = g_list[1].edata['feat']
        node_feats = g_list[1].ndata['feat']
        
        '''
        edge_feats = self.edge_rbf_path(edge_feats)
        edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
        edge_feats = self.Linear1(edge_feats)#2_path by RBF and linear to 64 
        
        node_feats = self.node_rbf_path(node_feats)
        node_feats = torch.tensor(node_feats, dtype=torch.float32)
        node_feats = self.Linear2(node_feats)
        node_feats = node_feats.to(device)
        node_feats = node_feats.to(edge_feats.dtype).to(device)
        
        edge_feats = edge_feats.to(device)
        '''
        
        ini_y3 = edge_feats.clone().detach()      
        
        ini_y2 = node_feats.clone()
    
        node_path = node_feats.clone()
        edge_path = edge_feats.clone()
          
        node_feats = g_list[0].ndata['feat']
        edge_feats = g_list[0].edata['feat']
  
        '''
        edge_feats = self.edge_rbf(edge_feats)
        edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
        edge_feats = self.Linear3(edge_feats)

        node_feats = self.node_rbf(node_feats)
        node_feats = torch.tensor(node_feats, dtype=torch.float32)
        
        node_feats = self.Linear4(node_feats)#0-path to 64
        node_feats = node_feats.to(edge_feats.dtype).to(device)
        '''
        
        ini_y1 = node_feats.clone().detach()
        
        # 在需要生成全零张量的地方
        
        out_1 = torch.zeros_like(node_feats).to(device)
        out_2 = torch.zeros_like(node_path).to(device)
        out_3 = torch.zeros_like(edge_path).to(device)
        for i in range(self.num_blcok):
            bolck = self.blockconv[i]
            node_feats,edge_feats,node_path,edge_path = bolck(g_list, device, node_feats,edge_feats,node_path,edge_path)

            if i == self.num_blcok-1:
                out_1 = node_feats
                out_2 = node_path
                out_3 = edge_path

        if resent:
            y_1 = (out_1 + ini_y1).to(torch.float32)
            y_2 = (out_2 + ini_y2).to(torch.float32)
            y_3 = (out_3 + ini_y3).to(torch.float32)
            
        else:
            y_1 = (out_1).to(torch.float32)
            y_2 = (out_2).to(torch.float32)
            y_3 = (out_3).to(torch.float32)
        
        if pooling == 'avg':

            y1 = F.adaptive_avg_pool1d(y_1.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
            y2 = F.adaptive_avg_pool1d(y_2.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
            y3 = F.adaptive_avg_pool1d(y_3.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        
        #y = torch.cat((y1, y3), dim=0)

        #y = torch.cat((out_1, out_2), dim=0)
        out = self.Readout(y1)
        return out


'''
class HL_GAT(nn.Module):
    
    #no RBF and information from high to low
    
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(HL_GAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.edge_feature_size = 10
        self.gcn_1_layers = nn.ModuleList()
        self.gcn_2_layers = nn.ModuleList()


        #第二层
        self.layer1 = EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2)

        self.gcn_1_layers.append(self.layer1)


        self.layer2= EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2)

        self.gcn_2_layers.append(self.layer2 )
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2))
        
        self.gate_nn = nn.Linear(self.atom_feature_size, 1)
        self.global_attention_pooling = GlobalAttentionPooling(self.gate_nn)
        self.sort_pooling = SortPooling(k=1)

        layers = [
            nn.Linear(self.atom_feature_size, out_feats),
            nn.Sigmoid(),
            nn.Linear(out_feats, 1),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout = nn.Sequential(*layers)

    def forward(self, g_list, device, resent, pooling):

        # 第二层GCN
        edge_feats = g_list[1].edata['feat'].to(device)
        node_feats = g_list[1].ndata['feat'].to(device)
        #edge_feats = (torch.tensor(self.edge_rbf_layer_2_path(edge_feats))).to(device)
        node_feats = node_feats.to(edge_feats.dtype).to(device)

        for layer in self.gcn_1_layers:
            node_feats, edge_feats = layer(g_list[1].to(device), node_feats, edge_feats)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)
        node_feats = node_feats.to(device)
        
        y_2 = node_feats
        
        m, n = y_2.shape

        # 使用 torch.cat 生成 2m x n 张量 B
        B = torch.cat([y_2[i:i+1].repeat(2, 1) for i in range(m)], dim=0)
        y_2 = B.to(device)


        #第三层
        node_feats = g_list[0].ndata['feat'].to(device)
        edge_feats = g_list[0].edata['feat'].to(device)
        #edge_feats = (torch.tensor(self.edge_rbf_layer(edge_feats))).to(device)
        
        concatenated_tensor = torch.cat((y_2.unsqueeze(0), edge_feats.unsqueeze(0)), dim=0)
        edge_feats = torch.add(concatenated_tensor[0], concatenated_tensor[1])

        node_feats = node_feats.to(edge_feats.dtype).to(device)
        ini_node_feat = F.adaptive_avg_pool1d(node_feats.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        
        for layer in self.gcn_2_layers:
            node_feats,edge_feats = layer(g_list[0].to(device), node_feats, edge_feats)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)

        y_11 = node_feats.to(device)    
        if pooling == 'glob':
            y_1 = self.global_attention_pooling(g_list[0].to(device),y_11).squeeze(0)
        if pooling == 'sort':
            y_1 = self.sort_pooling(g_list[0].to(device),y_11).squeeze(0)
            y_1 = y_1.squeeze(0)
        if pooling == 'avg':
            y_1 = F.adaptive_avg_pool1d(y_11.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        if resent == 'True':
            y = (y_1 + ini_node_feat).to(torch.float32)
            out = self.Readout(y)
        else:
            out = self.Readout(y_1)
        return out
    
'''



class LH_block(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(LH_block, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.edge_feature_size = 10
        self.gcn_1_layers = nn.ModuleList()
        self.gcn_2_layers = nn.ModuleList()
        
        #第二层
        self.layer1 = EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2) 

        self.gcn_1_layers.append(self.layer1)

        for _ in range(num_layers - 1):
            self.gcn_1_layers.append(EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2))


        self.linear_layer = nn.Linear(self.bond_feature_size*2, self.bond_feature_size)
        

        self.layer2= EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2)

        self.gcn_2_layers.append(self.layer2 )
        '''
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2))
        '''
        self.gate_nn = nn.Linear(self.atom_feature_size, 1)
        self.global_attention_pooling = GlobalAttentionPooling(self.gate_nn)
        self.sort_pooling = SortPooling(k=1)

        
    def forward(self, g_list, device, node_feats,edge_feats,node_path,edge_path):
        
        for layer in self.gcn_1_layers:
            node_feats, edge_feats = layer(g_list[0].to(device), node_feats, edge_feats)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)
        node_feats = node_feats.to(device)
        

        x_11 =  torch.cat((edge_feats[0::2, :], edge_feats[1::2, :]), dim=1) 
        node_path = self.linear_layer(x_11)
        
        #y_2 = node_path.to(device)


        #concatenated_tensor = torch.cat((y_2.unsqueeze(0), node_path.unsqueeze(0)), dim=0)
        #node_path = torch.add(concatenated_tensor[0], concatenated_tensor[1])

        
        for layer in self.gcn_2_layers:
            node_path,edge_path = layer(g_list[1].to(device), node_path, edge_path)
            node_path = node_path.to(device)
            edge_path = edge_path.to(device)
            node_path = torch.sum(node_path, dim=1)
            edge_path = torch.sum(edge_path, dim=1)
            
        return node_feats, edge_feats, node_path, edge_path    
        
        
        
        


class LH_GAT(nn.Module):
    '''
    no RBF information from low to high
    '''
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, num_blcok, dropout=0.2):
        super(LH_GAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_blcok = num_blcok
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.edge_feature_size = 10
        self.num_blcok = num_blcok
        self.blockconv = nn.ModuleList()

        
        '''
        self.edge_rbf_path = RBFKernelMappingLayer(input_dim=self.in_feats, output_dim=self.hidden_size, gamma=0.5)
        self.Linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.node_rbf_path = RBFKernelMappingLayer(input_dim=self.bond_feature_size, output_dim=self.hidden_size, gamma=0.5)
        self.Linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        #第三层
        self.node_rbf = RBFKernelMappingLayer(input_dim=self.atom_feature_size, output_dim=self.hidden_size, gamma=0.5)
        self.Linear3 = nn.Linear(self.hidden_size, self.hidden_size)

        self.edge_rbf = RBFKernelMappingLayer(input_dim=self.bond_feature_size, output_dim=self.hidden_size, gamma=0.5)
        self.Linear4 = nn.Linear(self.hidden_size, self.hidden_size)
        '''
        
        for i in range(self.num_blcok):
            self.blockconv.append(LH_block(in_feats, hidden_size, out_feats, encode_dim))
        
        self.relu = nn.ReLU()

        layers = [
            nn.Linear(self.in_feats, out_feats),
            nn.Sigmoid(),
            #NormalizeLayer(),
            nn.Dropout(dropout),
            nn.Linear(out_feats, 1),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout = nn.Sequential(*layers)

        

    def forward(self, g_list, device, resent, pooling):
    
        edge_feats = g_list[1].edata['feat']
        node_feats = g_list[1].ndata['feat']
        
        '''
        edge_feats = self.edge_rbf_path(edge_feats)
        edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
        edge_feats = self.Linear1(edge_feats)#2_path by RBF and linear to 64 
        
        node_feats = self.node_rbf_path(node_feats)
        node_feats = torch.tensor(node_feats, dtype=torch.float32)
        node_feats = self.Linear2(node_feats)
        node_feats = node_feats.to(device)
        node_feats = node_feats.to(edge_feats.dtype).to(device)
        
        edge_feats = edge_feats.to(device)
        '''
        
        ini_y3 = edge_feats.clone().detach()
        
        
        ini_y2 = node_feats.clone()
    
        node_path = node_feats.clone()
        edge_path = edge_feats.clone()
          
        node_feats = g_list[0].ndata['feat']
        edge_feats = g_list[0].edata['feat']
    
        '''
        edge_feats = self.edge_rbf(edge_feats)
        edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
        edge_feats = self.Linear3(edge_feats)
    
        node_feats = self.node_rbf(node_feats)
        node_feats = torch.tensor(node_feats, dtype=torch.float32)
        
        node_feats = self.Linear4(node_feats)#0-path to 64
        node_feats = node_feats.to(edge_feats.dtype).to(device)
        '''
        
        ini_y1 = node_feats.clone().detach()
        
        # 在需要生成全零张量的地方
        
        out_1 = torch.zeros_like(node_feats).to(device)
        out_2 = torch.zeros_like(node_path).to(device)
        out_3 = torch.zeros_like(edge_path).to(device)
        for i in range(self.num_blcok):
            bolck = self.blockconv[i]
            node_feats,edge_feats,node_path,edge_path = bolck(g_list, device, node_feats,edge_feats,node_path,edge_path)
    
            if i == self.num_blcok-1:
                out_1 = node_feats
                out_2 = node_path
                out_3 = edge_path
    
    
        if resent:
            y_1 = (out_1 + ini_y1).to(torch.float32)
            y_2 = (out_2 + ini_y2).to(torch.float32)
            y_3 = (out_3 + ini_y3).to(torch.float32)
            
        else:
            y_1 = (out_1).to(torch.float32)
            y_2 = (out_2).to(torch.float32)
            y_3 = (out_3).to(torch.float32)
        
        if pooling == 'avg':
    
    
            y1 = F.adaptive_avg_pool1d(y_1.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
            y2 = F.adaptive_avg_pool1d(y_2.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
            y3 = F.adaptive_avg_pool1d(y_3.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        
        #y = torch.cat((y1, y3), dim=0)
    
        #y = torch.cat((out_1, out_2), dim=0)
        out = self.Readout(y3)
        return out
    
'''

class LH_GAT(nn.Module):
    
    #no RBF information from low to high
    
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(LH_GAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.edge_feature_size = 10
        self.gcn_1_layers = nn.ModuleList()
        self.gcn_2_layers = nn.ModuleList()
        

        #第二层
        self.layer1 = EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2) 

        self.gcn_1_layers.append(self.layer1)


        self.linear_layer = nn.Linear(self.bond_feature_size*2, self.bond_feature_size)
        

        self.layer2= EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2)

        self.gcn_2_layers.append(self.layer2 )
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2))

        self.gate_nn = nn.Linear(self.atom_feature_size, 1)
        self.global_attention_pooling = GlobalAttentionPooling(self.gate_nn)
        self.sort_pooling = SortPooling(k=1)

        

        layers = [
            nn.Linear(self.in_feats, out_feats),
            nn.Sigmoid(),
            nn.Linear(out_feats, 1),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout = nn.Sequential(*layers)

    def forward(self, g_list, device, resent, pooling):

        # 第二层GCN
        edge_feats = g_list[0].edata['feat'].to(device)
        node_feats = g_list[0].ndata['feat'].to(device)
        #edge_feats = (torch.tensor(self.edge_rbf_layer_2_path(edge_feats))).to(device)
        node_feats = node_feats.to(edge_feats.dtype).to(device)

        for layer in self.gcn_1_layers:
            node_feats, edge_feats = layer(g_list[0].to(device), node_feats, edge_feats)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)
        node_feats = node_feats.to(device)
        

        x_11 =  torch.cat((edge_feats[0::2, :], edge_feats[1::2, :]), dim=1) 
        node_path = self.linear_layer(x_11)
        
        y_2 = node_path.to(device)


        #第三层
        node_feats = g_list[1].ndata['feat'].to(device)
        edge_feats = g_list[1].edata['feat'].to(device)


        concatenated_tensor = torch.cat((y_2.unsqueeze(0), node_feats.unsqueeze(0)), dim=0)
        node_feats = torch.add(concatenated_tensor[0], concatenated_tensor[1])

        
        ini_node_feat = F.adaptive_avg_pool1d(edge_feats.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        
        for layer in self.gcn_2_layers:
            node_feats,edge_feats = layer(g_list[1].to(device), node_feats, edge_feats)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)

        y_11 = edge_feats.to(device)  
         
        if pooling == 'glob':
            y_1 = self.global_attention_pooling(g_list[1].to(device),y_11).squeeze(0)
        if pooling == 'sort':
            y_1 = self.sort_pooling(g_list[1].to(device),y_11).squeeze(0)
            y_1 = y_1.squeeze(0)
        if pooling == 'avg':
            y_1 = F.adaptive_avg_pool1d(y_11.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        if resent:
            y = (y_1 + ini_node_feat).to(torch.float32)
            out = self.Readout(y)
        else:
            out = self.Readout(y_1)
        return out
'''




class LHL_block(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(LHL_block, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.gcn_1_layers = nn.ModuleList()
        self.gcn_2_layers = nn.ModuleList()
        self.gcn_3_layers = nn.ModuleList()


        #第二层
        self.layer1 = EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2) 

        self.gcn_1_layers.append(self.layer1)

        self.linear_layer_1 = nn.Linear(self.bond_feature_size*2, self.bond_feature_size)
        

        self.layer2= EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2)

        self.gcn_2_layers.append(self.layer2 )
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2))
            
        
        self.linear_layer_2 = nn.Linear(self.bond_feature_size*2, self.bond_feature_size)
        
        self.layer3 = EGATlayer(in_node_feats=self.atom_feature_size, in_edge_feats=self.bond_feature_size, out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size, num_heads=2) 
        self.gcn_3_layers.append(self.layer3)
        
        
        
    def forward(self, g_list, device, node_feats,edge_feats,node_path,edge_path):
        
        self.to(device)
        node_feats = node_feats.to(device)
        edge_feats = edge_feats.to(device)
        node_path = node_path.to(device)
        edge_path = edge_path.to(device)

        ini_y3 = node_path.clone()
        
        #L_information to H_information
        
        for layer in self.gcn_1_layers:
            node_feats, edge_feats = layer(g_list[0].to(device), node_feats, edge_feats)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)
        
        x_11 =  torch.cat((edge_feats[0::2, :], edge_feats[1::2, :]), dim=1) 
        node_path = self.linear_layer_1(x_11)
        
        y_2 = node_path.to(device)
                    
        concatenated_tensor = torch.cat((y_2.unsqueeze(0), ini_y3.unsqueeze(0)), dim=0)
        node_path = torch.add(concatenated_tensor[0], concatenated_tensor[1])
        
        
        #H_information to H_information
        for layer in self.gcn_2_layers:
            node_path,edge_path = layer(g_list[1].to(device), node_path, edge_path)
            node_path = node_path.to(device)
            edge_path = edge_path.to(device)
            node_path = torch.sum(node_path, dim=1)
            edge_path = torch.sum(edge_path, dim=1)
            
        
        m, n = node_path.shape

        # 使用 torch.cat 生成 2m x n 张量 B
        B = torch.cat([node_path[i:i+1].repeat(2, 1) for i in range(m)], dim=0)
        edge_feats = B.to(device)
        

        for layer in self.gcn_3_layers:
            node_feats, edge_feats = layer(g_list[0].to(device), node_feats, edge_feats)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)
        
        return node_feats,edge_feats,node_path,edge_path
    





class LHL_GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, num_blcok=1):
        super(LHL_GAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.encode_dim = encode_dim
        self.out_feats = out_feats
        self.num_blcok = num_blcok
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.blockconv = nn.ModuleList()
        
        '''
        self.edge_rbf_path = RBFKernelMappingLayer(input_dim=self.in_feats, output_dim=self.hidden_size, gamma=0.5)
        self.Linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.node_rbf_path = RBFKernelMappingLayer(input_dim=self.bond_feature_size, output_dim=self.hidden_size, gamma=0.5)
        self.Linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        #第三层
        self.node_rbf = RBFKernelMappingLayer(input_dim=self.atom_feature_size, output_dim=self.hidden_size, gamma=0.5)
        self.Linear3 = nn.Linear(self.hidden_size, self.hidden_size)

        self.edge_rbf = RBFKernelMappingLayer(input_dim=self.bond_feature_size, output_dim=self.hidden_size, gamma=0.5)
        self.Linear4 = nn.Linear(self.hidden_size, self.hidden_size)
        '''
        
        for i in range(self.num_blcok):
            self.blockconv.append(LHL_block(self.in_feats, self.hidden_size, self.out_feats, self.encode_dim))
        
        self.relu = nn.ReLU()

        layers = [
            nn.Linear(self.atom_feature_size, out_feats),
            nn.Sigmoid(),
            #NormalizeLayer(),
            nn.Linear(out_feats, 1),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout = nn.Sequential(*layers)
        
        
    def forward(self, g_list, device, resent, pooling):
        
        self.to(device)  # 将模型移动到 GPU 上

        edge_feats = g_list[1].edata['feat']
        node_feats = g_list[1].ndata['feat']
        
        '''
        node_feats = self.node_rbf_path(node_feats)
        node_feats = torch.tensor(node_feats, dtype=torch.float32)
        node_feats = self.Linear2(node_feats)
        node_feats = node_feats.to(device)
        node_feats = node_feats.to(edge_feats.dtype).to(device)
        
        
        edge_feats = self.edge_rbf_path(edge_feats)
        edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
        edge_feats = self.Linear1(edge_feats)#2_path by RBF and linear to 64 
        '''
        
        edge_feats = edge_feats.to(device)

        ini_y3 = edge_feats.clone().detach()
        
        ini_y2 = node_feats.clone().detach()
    
        node_path = node_feats
        edge_path = edge_feats
          
        node_feats = g_list[0].ndata['feat']
        edge_feats = g_list[0].edata['feat']
  
        '''
        edge_feats = self.edge_rbf(edge_feats)
        edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
        edge_feats = self.Linear3(edge_feats)

        node_feats = self.node_rbf(node_feats)
        node_feats = torch.tensor(node_feats, dtype=torch.float32)
        
        node_feats = self.Linear4(node_feats)#0-path to 64
        '''
        
        node_feats = node_feats.to(edge_feats.dtype).to(device)
        
        ini_y1 = node_feats.clone().detach()
        
        # 在需要生成全零张量的地方
        
        out_1 = torch.zeros_like(node_feats).to(device)
        out_2 = torch.zeros_like(node_path).to(device)
        out_3 = torch.zeros_like(edge_path).to(device)
        for i in range(self.num_blcok):
            bolck = self.blockconv[i]
            node_feats,edge_feats,node_path,edge_path = bolck(g_list, device, node_feats,edge_feats,node_path,edge_path)

            if i == self.num_blcok-1:
                out_1 = node_feats
                out_2 = node_path
                out_3 = edge_path
            '''
            if i == self.num_blcok-1:
                out_1 = out_1 + node_feats * 1/self.num_blcok
                out_2 = out_2 + node_path * 1/self.num_blcok
                out_3 = out_3 + edge_path * 1/self.num_blcok
            
            if i == self.num_blcok-1:
                out_1 += node_feats 
                out_2 += node_path 
                out_3 += edge_path 
            '''

        if resent:
            y_1 = (out_1 + ini_y1).to(torch.float32)
            y_2 = (out_2 + ini_y2).to(torch.float32)
            y_3 = (out_3 + ini_y3).to(torch.float32)
            
        else:
            y_1 = (out_1).to(torch.float32)
            y_2 = (out_2).to(torch.float32)
            y_3 = (out_3).to(torch.float32)
        
        if pooling == 'avg':

            y1 = F.adaptive_avg_pool1d(y_1.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
            y2 = F.adaptive_avg_pool1d(y_2.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
            y3 = F.adaptive_avg_pool1d(y_3.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        
        #y = torch.cat((y1), dim=0)

        #y = torch.cat((out_1, out_2), dim=0)
        out = self.Readout(y1)
        return out






class HLH_block(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(HLH_block, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.gcn_1_layers = nn.ModuleList()
        self.gcn_2_layers = nn.ModuleList()
        self.gcn_3_layers = nn.ModuleList()
        
        
        #第二层
        self.layer1 = EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2)

        self.gcn_1_layers.append(self.layer1)
        
        self.linear_layer_1 = nn.Linear(self.bond_feature_size*2, self.bond_feature_size)
        
        self.layer2= EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2)

        self.gcn_2_layers.append(self.layer2 )
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2))
     
        self.linear_layer_2 = nn.Linear(self.bond_feature_size*2, self.bond_feature_size)

        self.layer3 = EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2)
        self.gcn_3_layers.append(self.layer3)

        self.apply(initialize_weights)
        self.to(device)

    def forward(self, g_list, device, node_feats,edge_feats,node_path,edge_path):
        # 将模型的参数和输入数据移到 GPU 上
        self.to(device)
        node_feats = node_feats
        edge_feats = edge_feats
        node_path = node_path
        edge_path = edge_path

        ini_y3 = edge_feats.clone().detach()
        
        for layer in self.gcn_1_layers:
            g_list[1] = g_list[1].to(device)
            node_path, edge_path = layer(g_list[1], node_path, edge_path)
            node_path = torch.sum(node_path, dim=1)
            edge_path = torch.sum(edge_path, dim=1)
        
        y_2 = node_path.clone()
        m, n = y_2.shape

        # 使用 torch.cat 生成 2m x n 张量 B
        edge_feats = torch.cat([y_2[i:i+1].repeat(2, 1) for i in range(m)], dim=0).to(device)

        #第三层
        
        for i in range(self.num_layers):
            layer = self.gcn_2_layers[i]
            g_list[0] = g_list[0].to(device)
            node_feats,edge_feats = layer(g_list[0], node_feats, edge_feats)
            node_feats = torch.sum(node_feats, dim=1)
            edge_feats = torch.sum(edge_feats, dim=1)

        x_11 =  torch.cat((edge_feats[0::2, :], edge_feats[1::2, :]), dim=1) 
        node_path = self.linear_layer_2(x_11)

        for layer in self.gcn_3_layers:
            g_list[1] = g_list[1].to(device)
            node_path, edge_path = layer(g_list[1], node_path, edge_path)
            node_path = torch.sum(node_path, dim=1)
            edge_path = torch.sum(edge_path, dim=1)

        return node_feats,edge_feats,node_path,edge_path
    


class NormalizeLayer(nn.Module):
    def forward(self, x):
        normalized_tensor = F.normalize(x.view(1, -1), p=2, dim=1).squeeze()
        return normalized_tensor



class HLH_GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, num_blcok,dropout=0.2):
        super(HLH_GAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_blcok = num_blcok
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.blockconv = nn.ModuleList()
        
        self.edge_rbf_path = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.in_feats, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            #nn.LeakyReLU(),
            #nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.node_rbf_path = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.bond_feature_size, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            #nn.LeakyReLU(),
            #nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.node_rbf = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.atom_feature_size, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            #nn.LeakyReLU(),
            #nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.edge_rbf = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.bond_feature_size, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            #nn.LeakyReLU(),
            #nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        for i in range(self.num_blcok):
            self.blockconv.append(HLH_block(in_feats=64, hidden_size=64, out_feats=64, encode_dim=[64,64]))
        

        self.leaky_relu = nn.LeakyReLU()

        self.batch_norm = nn.BatchNorm1d(self.hidden_size*3)

        layers = [
            nn.Linear(self.hidden_size*3, out_feats),
            nn.Sigmoid(),
            #NormalizeLayer(),
            nn.Dropout(dropout),
            nn.Linear(out_feats, 1),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout = nn.Sequential(*layers)
        self.Readout.apply(initialize_weights)
        self.apply(initialize_weights)
        self.to(device)  # 将模型移动到 GPU 上

    def forward(self, batch_g_list, device, resent, pooling):
        # 将模型的参数和输入数据移到 GPU 上
        self.to(device)
        batch_outputs = []

        for g_list in batch_g_list:
            edge_path = g_list[1].edata['feat'].to(device)
            node_path = g_list[1].ndata['feat'].to(device)
            
            edge_path = self.edge_rbf_path(edge_path) 
            node_path = self.node_rbf_path(node_path)

            node_path = node_path.to(edge_path.dtype).to(device)
    
            ini_y3 = self.leaky_relu(edge_path).to(device)
            ini_y2 = self.leaky_relu(node_path).to(device)

            
            node_feats = g_list[0].ndata['feat'].to(device)
            edge_feats = g_list[0].edata['feat'].to(device)

            edge_feats = self.edge_rbf(edge_feats)
            node_feats = self.node_rbf(node_feats)
        
            ini_y1 = self.leaky_relu(node_feats)

            # 在需要生成全零张量的地方
            
            out_1 = torch.zeros_like(node_feats).to(device)
            out_2 = torch.zeros_like(node_path).to(device)
            out_3 = torch.zeros_like(edge_path).to(device)
            for i in range(self.num_blcok):
                bolck = self.blockconv[i]
                node_feats,edge_feats,node_path,edge_path = bolck(g_list, device, node_feats,edge_feats,node_path,edge_path)

                if i == 0:
                    out_1 = out_1 + node_feats.to(device)
                    out_2 = out_2 + node_path.to(device)
                    out_3 = out_3 + edge_path.to(device)

                if i == self.num_blcok-1:
                    out_1 = out_1 + node_feats.to(device)
                    out_2 = out_2 + node_path.to(device)
                    out_3 = out_3 + edge_path.to(device)


            if resent:
                y_1 = (out_1 + ini_y1).to(torch.float32).to(device)
                y_2 = (out_2 + ini_y2).to(torch.float32).to(device)
                y_3 = (out_3 + ini_y3).to(torch.float32).to(device)
                
            else:
                y_1 = (out_1).to(torch.float32).to(device)
                y_2 = (out_2).to(torch.float32).to(device)
                y_3 = (out_3).to(torch.float32).to(device)
            
            if pooling == 'avg':

                y1 = F.adaptive_avg_pool1d(y_1.unsqueeze(0).permute(0, 2, 1), 1).squeeze().to(device)
                y2 = F.adaptive_avg_pool1d(y_2.unsqueeze(0).permute(0, 2, 1), 1).squeeze().to(device)
                y3 = F.adaptive_avg_pool1d(y_3.unsqueeze(0).permute(0, 2, 1), 1).squeeze().to(device)
            
            y = torch.cat((y1, y2, y3), dim=0).to(device)
            batch_outputs.append(y)
        batch_out = torch.stack(batch_outputs, dim=0).to(device)
        #batch_out = self.batch_norm(batch_out)
        out = self.Readout(batch_out)
        return out
    
    


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)



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
        self.layer1 = EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2)

        self.gcn_1_layers.append(self.layer1)

        self.linear = nn.Linear(self.bond_feature_size*2,self.bond_feature_size)

        self.layer2= EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2)

        self.gcn_2_layers.append(self.layer2 )
        
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2))

        self.apply(initialize_weights)


    def forward(self, g_list, device, node_feats,edge_feats,node_path,edge_path):
        

        # from H information to L information
        for layer in self.gcn_1_layers:
            node_path, edge_path = layer(g_list[1].to(device), node_path, edge_path)
            node_path = node_path.to(device)
            edge_path = edge_path.to(device)
            node_path = torch.sum(node_path, dim=1)
            edge_path = torch.sum(edge_path, dim=1)
        node_path = node_path.to(device)
        
        y_2 = node_path
        
        m, n = y_2.shape

        # 使用 torch.cat 生成 2m x n 张量 B
        B = torch.cat([y_2[i:i+1].repeat(2, 1) for i in range(m)], dim=0)
        edge_feats = B.to(device)


        #concatenated_tensor = torch.cat((y_2, edge_feats), dim=1)
        #edge_feats = self.linear(concatenated_tensor)

        #concatenated_tensor = torch.cat((y_2, edge_feats), dim=0)
        #edge_feats = torch.add(concatenated_tensor[0], concatenated_tensor[1])

        node_feats = node_feats.to(edge_feats.dtype).to(device)

        
        for layer in self.gcn_2_layers:
            node_feats,edge_feats = layer(g_list[0].to(device), node_feats, edge_feats)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
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
        self.layer3 = EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2) 

        self.gcn_3_layers.append(self.layer3)
        for _ in range(num_layers - 1):
            self.gcn_3_layers.append(EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=2))
        
        self.linear_layer = nn.Linear(self.bond_feature_size*2,self.bond_feature_size)
        
        self.layer4= EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2)

        self.gcn_4_layers.append(self.layer4)
        '''
        for _ in range(num_layers - 1):
            self.gcn_4_layers.append(EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.in_feats,out_node_feats=self.bond_feature_size,out_edge_feats=self.in_feats,num_heads=2))
        '''

        self.apply(initialize_weights)

    def forward(self, g_list, device, node_feats,edge_feats,node_path,edge_path):
        

        for layer in self.gcn_3_layers:
            node_feats, edge_feats= layer(g_list[0].to(device), node_feats, edge_feats)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            node_feats= torch.sum(node_feats, dim=1)
            edge_feats= torch.sum(edge_feats, dim=1)
        node_feats = node_feats.to(device)
        

        x_11 =  torch.cat((edge_feats[0::2, :], edge_feats[1::2, :]), dim=1) 
        node_path = self.linear_layer(x_11)
        

        #y_2 = node_feats.to(device)
        #concatenated_tensor = torch.cat((y_2.unsqueeze(0), node_path.unsqueeze(0)), dim=0)
        #node_path = torch.add(concatenated_tensor[0], concatenated_tensor[1])
        
        for layer in self.gcn_4_layers:
            node_path, edge_path= layer(g_list[1].to(device), node_path, edge_path)
            node_path = node_path.to(device)
            edge_path = edge_path.to(device)
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


        self.layer2= HL_LH_block_2(in_feats, hidden_size, out_feats, encode_dim)

        self.block_layers.append(self.layer2)

        


    def forward(self, g_list, device, node_feats,edge_feats,node_path,edge_path):

        # from H information to L information
        for i in range(self.num_layers):
            layer = self.block_layers[i]
            if i == 0:
                node_feats, edge_feats = layer(g_list, device, node_feats,edge_feats,node_path,edge_path)
            
            if i == 1:
                node_path, edge_path = layer(g_list, device, node_feats,edge_feats,node_path,edge_path)


        return node_feats, edge_feats, node_path, edge_path
    
    



class HL_LH_GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_blcok=1):
        super(HL_LH_GAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_blcok = num_blcok
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.angle_feature_size = encode_dim[2]
        self.dihedral_feature_size = encode_dim[3]
        self.blockconv = nn.ModuleList()

        
        self.edge_rbf_path = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.in_feats, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.node_rbf_path = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.bond_feature_size, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )


        self.node_rbf = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.atom_feature_size, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.edge_rbf = nn.Sequential(
            RBFKernelMappingLayer(
                input_dim=self.bond_feature_size, 
                output_dim=self.hidden_size, 
                gamma=0.5),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.den_fg_edges = nn.Linear(self.dihedral_feature_size,self.hidden_size)
        self.den_fg_nodes = nn.Linear(self.angle_feature_size,self.hidden_size)

        self.den_lg_edges = nn.Linear(self.angle_feature_size,self.hidden_size)
        self.den_lg_nodes = nn.Linear(self.bond_feature_size,self.hidden_size)

        self.den_g_edges = nn.Linear(self.bond_feature_size,self.hidden_size)
        self.den_g_nodes = nn.Linear(self.atom_feature_size,self.hidden_size)

        '''

        self.edge_rbf_path = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=1.0,
                bins= self.in_feats,
            ),
            nn.Linear(self.in_feats, self.hidden_size),
            nn.Softplus(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )


        self.node_rbf_path = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=1.0,
                bins= self.bond_feature_size,
            ),
            nn.Linear(self.bond_feature_size, self.hidden_size),
            nn.Softplus(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        

        #第三层
        self.node_rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=1.0,
                bins= self.atom_feature_size,
            ),
            nn.Linear(self.atom_feature_size, self.hidden_size),
            nn.Softplus(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.edge_rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=10.0,
                bins= self.bond_feature_size,
            ),
            nn.Linear(self.bond_feature_size, self.hidden_size),
            nn.Softplus(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        '''

        for i in range(self.num_blcok):
            self.blockconv.append(HL_LH_block(in_feats=64, hidden_size=64, out_feats=64, encode_dim=[64,64]))
        
        self.Linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        
        self.leaky_relu = nn.LeakyReLU()

        layers = [
            nn.Linear(self.hidden_size*2, out_feats),
            self.leaky_relu,
            nn.Dropout(dropout),
            #nn.Sigmoid(),
            #NormalizeLayer(),
            nn.Linear(out_feats, 1),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout = nn.Sequential(*layers)
        self.Readout.apply(initialize_weights)
        self.apply(initialize_weights)
        
    def forward(self, g_list, device, resent, pooling):
        
        edge_path = g_list[1].edata['feat'].to(device)
        node_path = g_list[1].ndata['feat'].to(device)
        
        edge_path = self.edge_rbf_path(edge_path)
        #edge_path = self.Linear1(edge_path)#2_path by RBF and linear to 64 
        
        node_path = self.node_rbf_path(node_path)
        #node_path = self.Linear2(node_path)

        node_path = node_path.to(edge_path.dtype).to(device)
        edge_path = edge_path.to(device)
   
        ini_y3 = self.leaky_relu(edge_path)
        ini_y2 = self.leaky_relu(node_path)

          
        node_feats = g_list[0].ndata['feat']
        edge_feats = g_list[0].edata['feat']

        edge_feats = self.edge_rbf(edge_feats)
        #edge_feats = self.Linear3(edge_feats)

        node_feats = self.node_rbf(node_feats)
        #node_feats = self.Linear4(node_feats)#0-path to 64
        node_feats = node_feats.to(edge_feats.dtype).to(device)
      
        ini_y1 = self.leaky_relu(node_feats)
        
        # 在需要生成全零张量的地方
        
        out_1 = torch.zeros_like(node_feats).to(device)
        out_2 = torch.zeros_like(node_path).to(device)
        out_3 = torch.zeros_like(edge_path).to(device)
        for i in range(self.num_blcok):
            bolck = self.blockconv[i]
            node_feats,edge_feats,node_path,edge_path = bolck(g_list.to(device), device.to(device), node_feats.to(device),edge_feats.to(device),node_path.to(device),edge_path.to(device))

            if i == self.num_blcok-1:
                edge_embedd = torch.cat((edge_feats[0::2, :], edge_feats[1::2, :]), dim=1)
                out_1 = node_feats.to(device)
                out_2 = self.Linear(edge_embedd).to(device)
                out_3 = edge_path.to(device)


        if resent:
            y_1 = (out_1 + ini_y1).to(torch.float32).to(device)
            y_2 = (out_2 + ini_y2).to(torch.float32).to(device)
            y_3 = (out_3 + ini_y3).to(torch.float32).to(device)
            
        else:
            y_1 = (out_1).to(torch.float32).to(device)
            y_2 = (out_2).to(torch.float32).to(device)
            y_3 = (out_3).to(torch.float32).to(device)
        
        if pooling == 'avg':

            y1 = F.adaptive_avg_pool1d(y_1.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
            y2 = F.adaptive_avg_pool1d(y_2.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
            y3 = F.adaptive_avg_pool1d(y_3.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        
        y = torch.cat((y1, y3), dim=0)

        #y = torch.cat((out_1, out_2), dim=0)
        out = self.Readout(y)
        return out