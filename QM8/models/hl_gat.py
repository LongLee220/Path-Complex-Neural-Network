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



from typing import Optional
from torch.nn import init
from torch_geometric.nn import MessagePassing, global_mean_pool
from sklearn.kernel_approximation import RBFSampler
from dgl.nn import GlobalAttentionPooling, SortPooling
from dgl.utils import expand_as_pair 
from dgl.nn.functional import edge_softmax
from dgl import DGLError
from models.activations import swish
from models.envelope import Envelope
from models.initializers import GlorotOrthogonal
from dgl.nn import AvgPooling





device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")






def pool_subgraphs_node(out, batched_graph):
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


def pool_subgraphs_edge(out, batched_graph):
    # 将整图的输出按照子图数量拆分成子图的输出
    subgraphs = dgl.unbatch(batched_graph)

    # 对每个子图进行池化操作，这里使用了平均池化
    pooled_outputs = []
    ini = 0
    for subgraph in subgraphs:
        # 获取子图节点的数量
        num_edges = subgraph.num_edges()
        
        # 根据子图的节点数量对整图的输出进行切片
        start_idx = ini
        end_idx = start_idx + num_edges
        sg_out = out[start_idx:end_idx]
        ini += num_edges
        # 计算每个子图的平均池化表示
        #print(sg_out.shape)
        #pooled_out = F.avg_pool2d(sg_out, kernel_size=num_nodes)  # 使用节点数作为池化核的大小
        pooled_out = F.adaptive_avg_pool1d(sg_out.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        #pooled_out = F.adaptive_max_pool1d(sg_out.unsqueeze(0).permute(0, 2, 1), 1).squeeze()

        pooled_outputs.append(pooled_out)

    return torch.stack(pooled_outputs)





'''   
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
'''         



class EGATlayer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads, bias=True):
        super(EGATlayer, self).__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_node = nn.Linear(in_node_feats, out_node_feats * num_heads, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats * num_heads, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_node.weight)
        nn.init.xavier_normal_(self.fc_ni.weight)
        nn.init.xavier_normal_(self.fc_fij.weight)
        nn.init.xavier_normal_(self.fc_nj.weight)
        nn.init.xavier_normal_(self.attn)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, graph, nfeats, efeats, get_attention=False):
        with graph.local_scope():
            in_degrees = graph.in_degrees().float().unsqueeze(-1)
            in_degrees[in_degrees == 0] = 1  # 将入度为0的节点设置为1，以避免除零错误
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

            #reshaped_tensor = in_degrees.unsqueeze(1).repeat(1, self._num_heads, 1)
            #h_out = h_out/reshaped_tensor

            if get_attention:
                return h_out, f_out, graph.edata.pop('a')
            else:
                return h_out,f_out



class HL_LH_block_1(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, num_heads,num_layers):
        super(HL_LH_block_1, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.angle_feature_size = encode_dim[2]
        self.dihedral_feature_size = encode_dim[3]

        
        self.g_layers = nn.ModuleList()
        self.lg_layers = nn.ModuleList()
        self.fg_layers = nn.ModuleList()
        
        
        #H_information to L_information 
        self.fg_layer = EGATlayer(in_node_feats=self.angle_feature_size,in_edge_feats=self.dihedral_feature_size,out_node_feats=self.angle_feature_size,out_edge_feats=self.dihedral_feature_size,num_heads=self.num_heads)

        for _ in range(self.num_layers-self.num_layers):
            self.fg_layers.append(self.fg_layer)


        self.lg_layer = EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.angle_feature_size,out_node_feats=self.bond_feature_size,out_edge_feats=self.angle_feature_size,num_heads=self.num_heads*8)

        for _ in range(self.num_layers-1):
            self.lg_layers.append(self.lg_layer)


        self.g_layer = EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=self.num_heads*8)
        
        for _ in range(self.num_layers):
            self.g_layers.append(self.g_layer)

        self.fg_linear = nn.Linear(self.angle_feature_size*2,self.angle_feature_size)
        self.lg_linear = nn.Linear(self.bond_feature_size*2,self.bond_feature_size)

    def forward(self, g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat):
        

        # from H information to L information
        if len(self.fg_layers) > 0:
            for layer in self.fg_layers:
                fg_node_feat,fg_edge_feat = layer(fg.to(device), fg_node_feat,fg_edge_feat)
                fg_node_feat = torch.sum(fg_node_feat, dim=1)
                fg_edge_feat = torch.sum(fg_edge_feat, dim=1)

            #lg_edge_feat = self.fg_linear(torch.cat((fg_node_feat, lg_edge_feat), dim=1))
            lg_edge_feat = torch.add(fg_node_feat, lg_edge_feat)
            #lg_edge_feat = fg_node_feat

        
        if len(self.lg_layers) > 0:

            for layer in self.lg_layers:
                lg_node_feat, lg_edge_feat = layer(lg.to(device), lg_node_feat, lg_edge_feat)
                lg_node_feat = torch.sum(lg_node_feat, dim=1)
                lg_edge_feat = torch.sum(lg_edge_feat, dim=1)

            #g_edge_feat = self.lg_linear(torch.cat((lg_node_feat, g_edge_feat), dim=1))
            g_edge_feat = torch.add(lg_node_feat, g_edge_feat)
            #g_edge_feat = lg_node_feat
        
        for layer in self.g_layers:
            g_node_feat, g_edge_feat = layer(g.to(device), g_node_feat, g_edge_feat)
            g_node_feat = torch.sum(g_node_feat, dim=1)
            g_edge_feat = torch.sum(g_edge_feat, dim=1)


        return  g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat
            



class HL_LH_block_2(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(HL_LH_block_2, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.encode_dim = encode_dim
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.angle_feature_size = encode_dim[2]
        self.dihedral_feature_size = encode_dim[3]

        self.g_layers = nn.ModuleList()
        self.lg_layers = nn.ModuleList()
        self.fg_layers = nn.ModuleList()
        
        
        #L_informatin to H_ingromation 
        self.g_layer = EGATlayer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=8) 

        for _ in range(num_layers-num_layers):
            self.g_layers.append(self.g_layer)

        self.lg_layer = EGATlayer(in_node_feats=self.bond_feature_size,in_edge_feats=self.angle_feature_size,out_node_feats=self.bond_feature_size,out_edge_feats=self.angle_feature_size,num_heads=8) 


        for _ in range(num_layers):
            self.lg_layers.append(self.lg_layer)
        
        self.fg_layer = EGATlayer(in_node_feats=self.angle_feature_size,in_edge_feats=self.dihedral_feature_size,out_node_feats=self.angle_feature_size,out_edge_feats=self.dihedral_feature_size,num_heads=8)


        for _ in range(num_layers+1):
            self.fg_layers.append(self.fg_layer)


        self.fg_linear = nn.Linear(self.angle_feature_size*2,self.angle_feature_size)
        self.lg_linear = nn.Linear(self.bond_feature_size*2,self.bond_feature_size)

    def forward(self,g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat):
        

        for layer in self.g_layers:
            g_node_feat, g_edge_feat= layer(g.to(device),g_node_feat, g_edge_feat)
            g_node_feat = torch.sum(g_node_feat, dim=1)
            g_edge_feat = torch.sum(g_edge_feat, dim=1)
        

        #lg_node_feat = torch.add(lg_node_feat, g_edge_feat)
        #lg_node_feat = g_edge_feat
        lg_node_feat = self.lg_linear(torch.cat((lg_node_feat, g_edge_feat), dim=1))

        

        for layer in self.lg_layers:
            lg_node_feat, lg_edge_feat = layer(lg.to(device), lg_node_feat, lg_edge_feat)
            lg_node_feat = torch.sum(lg_node_feat, dim=1)
            lg_edge_feat = torch.sum(lg_edge_feat, dim=1)

        #fg_node_feat = torch.add(fg_node_feat, lg_edge_feat)
        #fg_node_feat = lg_edge_feat
        fg_node_feat = self.fg_linear(torch.cat((fg_node_feat, lg_edge_feat), dim=1))

        for layer in self.fg_layers:
            fg_node_feat, fg_edge_feat = layer(fg.to(device), fg_node_feat, fg_edge_feat)
            fg_node_feat = torch.sum(fg_node_feat, dim=1)
            fg_edge_feat = torch.sum(fg_edge_feat, dim=1)

        return g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat
    




class HL_LH_block(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, num_heads, num_layers):
        super(HL_LH_block, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.encode_dim = encode_dim
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.angle_feature_size = encode_dim[2]
        self.dihedral_feature_size = encode_dim[3]

        self.block_layers = nn.ModuleList()

        
        #H_information to L_information 
        self.hl_dir = HL_LH_block_1(in_feats = self.in_feats, hidden_size=self.hidden_size, out_feats=self.out_feats, encode_dim=self.encode_dim, num_heads=self.num_heads,num_layers=self.num_layers)

        self.lh_dir = HL_LH_block_2(in_feats, hidden_size, out_feats, encode_dim)

        self.block_layers.append(self.hl_dir)
        self.block_layers.append(self.lh_dir)

        


    def forward(self, g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat,tras_med):

        # from H information to L information

        if tras_med == 'hl':
            for i in range(len(self.block_layers)):
                layer = self.block_layers[i]
                if i == 0:
                    g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat = layer(g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat)

        if tras_med == 'lh':
            for i in range(len(self.block_layers)):
                layer = self.block_layers[i]
                if i == 1:
                    g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat = layer(g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat)

                
        if tras_med == 'hllh':
            for i in range(len(self.block_layers)):
                layer = self.block_layers[i]
                g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat = layer(g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat)

        if tras_med == 'lhhl':
            for i in range(len(self.block_layers)):
                layer = self.block_layers[1-i]
                g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat = layer(g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat)

        if tras_med == 'hl-lh':
            for i in range(len(self.block_layers)):
                layer = self.block_layers[i]
                if i == 0:
                    g_node_hl,g_edge_hl,lg_node_hl,lg_edge_hl,fg_node_hl,fg_edge_hl = layer(g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat)
                if i == 1:
                    g_node_lh,g_edge_lh,lg_node_lh,lg_edge_lh,fg_node_lh,fg_edge_lh = layer(g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat)

            g_node_feat = g_node_hl + g_node_lh
            g_edge_feat = g_edge_hl + g_edge_lh

            lg_node_feat = lg_node_hl + lg_node_lh
            lg_edge_feat = lg_edge_hl + lg_edge_lh

            fg_node_feat = fg_node_hl + fg_node_lh
            fg_edge_feat = fg_edge_hl + fg_edge_lh

        return g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat
    
    




class RBF_GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, out_dim, tras_med, num_blcok,num_heads,num_layers,activation=swish,dropout = 0.1):
        super(RBF_GAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_blcok = num_blcok
        self.tras_med = tras_med
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.activation = activation
        self.encode_dim = encode_dim
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.angle_feature_size = encode_dim[2]
        self.dihedral_feature_size = encode_dim[3]
        self.dropout = dropout

        self.blockconv = nn.ModuleList()

        self.den_fg_edges = nn.Linear(self.dihedral_feature_size,self.hidden_size)
        self.den_fg_nodes = nn.Linear(self.angle_feature_size,self.hidden_size)

        self.den_lg_edges = nn.Linear(self.angle_feature_size,self.hidden_size)
        self.den_lg_nodes = nn.Linear(self.bond_feature_size,self.hidden_size)

        self.den_g_edges = nn.Linear(self.bond_feature_size,self.hidden_size)
        self.den_g_nodes = nn.Linear(self.atom_feature_size,self.hidden_size)


        '''
        for i in range(self.num_blcok):
            self.blockconv.append(HL_LH_block(in_feats=64, hidden_size=64, out_feats=64, encode_dim=[64,64,64,64]))
        '''

        for i in range(self.num_blcok):
            self.blockconv.append(HL_LH_block(in_feats=self.in_feats, hidden_size=self.hidden_size, out_feats=self.hidden_size, encode_dim=self.encode_dim,num_heads=self.num_heads, num_layers=self.num_layers))
            

        self.leaky_relu = nn.LeakyReLU()

        self.norm = nn.BatchNorm1d(self.hidden_size*2)

        layers1 = [
            nn.Linear(self.hidden_size*3, out_feats),
            self.leaky_relu,
            #self.activation,
            nn.Linear(out_feats, self.out_dim),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout1 = nn.Sequential(*layers1)

        layers2 = [
            nn.Linear(self.hidden_size*6, self.hidden_size*2),
            self.leaky_relu,
            #self.activation,
            nn.Linear(self.hidden_size*2, self.out_dim),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout2 = nn.Sequential(*layers2)

        layers3 = [
            nn.Linear(sum(self.encode_dim), self.hidden_size),
            nn.Sigmoid(),
            #self.activation,
            nn.Linear(self.hidden_size, self.out_dim),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout3 = nn.Sequential(*layers3)

        layers4 = [
            nn.Linear(self.encode_dim[0]+self.encode_dim[1], self.hidden_size),
            nn.Sigmoid(),
            #nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.out_dim),
            nn.Sigmoid()
        ]
        self.Readout4 = nn.Sequential(*layers4)

        #self.Readout.apply(self.initialize_weights)
        #self.apply(self.initialize_weights)

    
    def reset_params(self):
        GlorotOrthogonal(self.den_fg_edges.weight)
        GlorotOrthogonal(self.den_fg_nodes.weight)

        GlorotOrthogonal(self.den_lg_edges.weight)
        GlorotOrthogonal(self.den_lg_nodes.weight)

        GlorotOrthogonal(self.den_g_edges.weight)
        GlorotOrthogonal(self.den_g_nodes.weight)


    def forward(self, g_feats, g, lg, fg, device, resent, pooling):
        
        #print(g, lg, fg)
        
        batch_outputs = []

        '''
        g_graph = g.to(device) 
        g_graph_node_feat = self.den_g_nodes(g_graph.ndata['feat'])
        g_graph_edge_feat = self.den_g_edges(g_graph.edata['feat'])
        ini_y1 = self.activation(g_graph_node_feat)


        lg_graph = lg.to(device) 
        lg_graph_node_feat = self.den_lg_nodes(lg_graph.ndata['feat'])
        lg_graph_edge_feat = self.den_lg_edges(lg_graph.edata['feat'])
        ini_y2 = self.activation(lg_graph_node_feat)


        fg_graph = fg.to(device) 
        fg_graph_node_feat = self.den_fg_nodes(fg_graph.ndata['feat'])
        fg_graph_edge_feat = self.den_fg_edges(fg_graph.edata['feat'])
        ini_y3 = self.activation(fg_graph_node_feat)
        
        '''
        g_graph = g.to(device) 
        g_graph_node_feat = g_graph.ndata['feat']
        g_graph_edge_feat = g_graph.edata['feat']
        #ini_y1 = self.activation(g_graph_node_feat)
        ini_y1 = g_graph_node_feat


        lg_graph = lg.to(device) 
        lg_graph_node_feat = lg_graph.ndata['feat']
        lg_graph_edge_feat = lg_graph.edata['feat']
        #ini_y2 = self.activation(lg_graph_node_feat)
        ini_y2 = lg_graph_node_feat


        fg_graph = fg.to(device) 
        fg_graph_node_feat = fg_graph.ndata['feat']
        fg_graph_edge_feat = fg_graph.edata['feat']
        ini_y3 = fg_graph_node_feat
        
        ini_y4 = fg_graph_edge_feat
        # 在需要生成全零张量的地方
        
        out_1 = torch.zeros_like(g_graph_node_feat)
        out_2 = torch.zeros_like(lg_graph_node_feat)
        out_3 = torch.zeros_like(fg_graph_node_feat)
        out_4 = torch.zeros_like(fg_graph_edge_feat)

        for i in range(self.num_blcok):
            bolck = self.blockconv[i]
            g_graph_node_feat,g_graph_edge_feat, lg_graph_node_feat, lg_graph_edge_feat, fg_graph_node_feat,fg_graph_edge_feat = bolck(g,lg,fg, device, g_graph_node_feat,g_graph_edge_feat, lg_graph_node_feat, lg_graph_edge_feat, fg_graph_node_feat,fg_graph_edge_feat,self.tras_med)


            out_1 = out_1 + g_graph_node_feat
            out_2 = out_2 + lg_graph_node_feat
            out_3 = out_3 + fg_graph_node_feat
            out_4 = out_4 + fg_graph_edge_feat


        if resent:
            '''
            out_1 = torch.cat((out_1, ini_y1), dim=1)
            out_2 = torch.cat((out_2, ini_y2), dim=1)
            out_3 = torch.cat((out_3, ini_y3), dim=1)
            out_4 = torch.cat((out_4, ini_y3), dim=1)
            '''

            out_1 = out_1 + ini_y1
            out_2 = out_2 + ini_y2
            out_3 = out_3 + ini_y3
            out_4 = out_4 + ini_y4


        if pooling == 'avg':
            #y1 = AvgPooling(out_1, g)
            #y2 = AvgPooling(out_2, lg)
            #y3 = AvgPooling(out_3, fg)

            y1 = pool_subgraphs_node(out_1, g)
            y2 = pool_subgraphs_node(out_2, lg)
            y3 = pool_subgraphs_node(out_3, fg)
            y4 = pool_subgraphs_edge(out_4, fg)

        #y5 = g_feats.to(device)
        #y = torch.cat((y1, y2, y3, y4, y5), dim=1)
            
        #y = torch.cat((y1, y2, y3, y4), dim=1)
        y = torch.cat((y1,y2), dim=1)
        #y = y1
        batch_outputs.append(y)
        batch_out = y.squeeze() 

        out = self.Readout4(batch_out)
        #print(batch_out.shape)
        #out = self.norm(batch_out)
        
        return out
    
