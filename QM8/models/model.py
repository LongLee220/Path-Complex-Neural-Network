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
        self.RBF = RBFSampler(gamma = self.gamma, n_components = self.output_dim)

    def forward(self, input_tensor):
        mapped_tensor = self.RBF.fit_transform(input_tensor)
        return mapped_tensor



class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.in_feats = in_feats
        self.output_dim = out_feats
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, node_feats, edge_feats=None):

        # 确保数据类型匹配或进行转换
        if edge_feats is not None:
            edge_feats = edge_feats.to(node_feats.dtype)

        with g.local_scope():
            g.ndata['h'] = node_feats
            if edge_feats is not None:
                g.edata['e'] = edge_feats
                # 聚合邻居节点信息，同时考虑边的特征
                g.update_all(fn.u_add_e('h', 'e', 'm'), fn.mean('m', 'neigh'))
                # 使用边的特征来更新节点表示
                combined_feats = torch.cat([g.ndata['neigh'], g.ndata['h']], dim=1)
                combined_feats = combined_feats.to(self.linear.weight.dtype)
                node_feats = self.linear(combined_feats)
            else:
                # 仅考虑邻居节点信息
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
                node_feats = torch.relu(self.linear(g.ndata['neigh']))

        return node_feats


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.edge_feature_size = 1
        self.gcn_1_layers = nn.ModuleList()
        self.gcn_2_layers = nn.ModuleList()

        # RBF核函数的参数

        #第二层
        self.edge_rbf_layer_2_path = RBFKernelMappingLayer(input_dim=1, output_dim=self.bond_feature_size, gamma=0.1)
        #第二层
        self.layer1 = GCNLayer(self.bond_feature_size*2, self.atom_feature_size)
        self.gcn_1_layers.append(self.layer1)

    
        #第三层
        self.edge_rbf_layer = RBFKernelMappingLayer(input_dim=self.bond_feature_size, output_dim=self.atom_feature_size, gamma=0.1)

        self.layer2= GCNLayer(self.atom_feature_size*2, self.atom_feature_size)
        self.gcn_2_layers.append(self.layer2 )
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(GCNLayer(self.atom_feature_size*2, self.atom_feature_size))
        
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
        edge_feats = g_list[1].edata['feat']
        node_feats = g_list[1].ndata['feat'].to(device)
        edge_feats = (torch.tensor(self.edge_rbf_layer_2_path(edge_feats))).to(device)
        node_feats = node_feats.to(edge_feats.dtype).to(device)

        for layer in self.gcn_1_layers:

            node_feats = layer(g_list[1].to(device), node_feats, edge_feats).to(device)
      
        node_feats = node_feats.to(device)
        y_2 = node_feats
        m, n = y_2.shape

        # 使用 torch.cat 生成 2m x n 张量 B
        B = torch.cat([y_2[i:i+1].repeat(2, 1) for i in range(m)], dim=0)
        y_2 = B.to(device)


        #第三层
        node_feats = g_list[0].ndata['feat']
        edge_feats = g_list[0].edata['feat']
        edge_feats = (torch.tensor(self.edge_rbf_layer(edge_feats))).to(device)
        
        concatenated_tensor = torch.cat((y_2.unsqueeze(0), edge_feats.unsqueeze(0)), dim=0)
        edge_feats = torch.add(concatenated_tensor[0], concatenated_tensor[1])

        node_feats = node_feats.to(edge_feats.dtype).to(device)
        ini_node_feat = F.adaptive_avg_pool1d(node_feats.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        
        for layer in self.gcn_2_layers:

            node_feats = layer(g_list[0].to(device), node_feats, edge_feats).to(device)
            
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
    


class GINElayer(nn.Module):
    def __init__(self,apply_func=None,init_eps=0,learn_eps=False):
        super(GINElayer, self).__init__()
        self.apply_func = apply_func
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def message(self, edges):
        return {'m': F.relu(edges.src['hn'] + edges.data['he'])}

    def forward(self, graph, node_feat, edge_feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(node_feat, graph)
            graph.srcdata['hn'] = feat_src
            graph.edata['he'] = edge_feat
            graph.update_all(self.message, fn.sum('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst
        


class GIN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_layers=2):
        super(GIN, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.edge_feature_size = 1
        self.gcn_1_layers = nn.ModuleList()
        self.gcn_2_layers = nn.ModuleList()

        # RBF核函数的参数

        #第二层
        self.edge_rbf_layer_2_path = RBFKernelMappingLayer(input_dim=1, output_dim=self.bond_feature_size, gamma=0.1)
        #第二层
        self.layer1 = GINElayer(nn.Linear(self.bond_feature_size,self.atom_feature_size))
        self.gcn_1_layers.append(self.layer1)

    
        #第三层
        self.edge_rbf_layer = RBFKernelMappingLayer(input_dim=self.bond_feature_size, output_dim=self.atom_feature_size, gamma=0.1)

        self.layer2= GINElayer()
        self.gcn_2_layers.append(self.layer2 )
        for _ in range(num_layers - 1):
            self.gcn_2_layers.append(GINElayer())
        
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
        edge_feats = g_list[1].edata['feat']
        node_feats = g_list[1].ndata['feat'].to(device)
        edge_feats = (torch.tensor(self.edge_rbf_layer_2_path(edge_feats))).to(device)
        node_feats = node_feats.to(edge_feats.dtype).to(device)

        for layer in self.gcn_1_layers:
            node_feats = layer(g_list[1].to(device), node_feats, edge_feats).to(device)
      
        node_feats = node_feats.to(device)
        y_2 = node_feats
        m, n = y_2.shape

        # 使用 torch.cat 生成 2m x n 张量 B
        B = torch.cat([y_2[i:i+1].repeat(2, 1) for i in range(m)], dim=0)
        y_2 = B.to(device)


        #第三层
        node_feats = g_list[0].ndata['feat']
        edge_feats = g_list[0].edata['feat']
        edge_feats = (torch.tensor(self.edge_rbf_layer(edge_feats))).to(device)
        
        concatenated_tensor = torch.cat((y_2.unsqueeze(0), edge_feats.unsqueeze(0)), dim=0)
        edge_feats = torch.add(concatenated_tensor[0], concatenated_tensor[1])

        node_feats = node_feats.to(edge_feats.dtype).to(device)
        ini_node_feat = F.adaptive_avg_pool1d(node_feats.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        
        for layer in self.gcn_2_layers:

            node_feats = layer(g_list[0].to(device), node_feats, edge_feats).to(device)
            
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
    def __init__(self, in_feats, hidden_size, out_feats,output, encode_dim, dropout=0.2, num_layers=2):
        super(EGAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.edge_feature_size = 1
        self.output = output
        self.gcn_1_layers = nn.ModuleList()
        self.gcn_2_layers = nn.ModuleList()

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

        layers = [
            nn.Linear(self.atom_feature_size, out_feats),
            nn.Sigmoid(),
            nn.Linear(out_feats, self.output),
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
    




class GNNet(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, dropout=0.2, num_net=4, file=None):
        super(GNNet, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats

        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]

        self.edge_rbf_layer_2_path = RBFKernelMappingLayer(input_dim=self.in_feats, output_dim=self.bond_feature_size, gamma=0.1)

        self.edge_rbf_layer = RBFKernelMappingLayer(input_dim=self.bond_feature_size, output_dim=self.atom_feature_size, gamma=0.1)


        self.num_net = num_net
        self.drug1_gcn1 = nn.ModuleList([dglnn.GraphConv(self.bond_feature_size, self.hidden_size, norm='both').double() for i in range(num_net)])

        self.drug1_gcn2 = nn.ModuleList([dglnn.GraphConv(self.atom_feature_size,self.hidden_size, norm='both').double() for i in range(num_net)])
        
        # combined layers
        self.fc1 = nn.Linear(self.hidden_size, 64).double()
        self.out = nn.Linear(64, self.out_feats).double()

        # activation and regularization
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g_list, device, resent, pooling):

        x11 = torch.Tensor().to(device)

        node_feats = g_list[1].ndata['feat'].to(device)
        for layer in self.drug1_gcn1:
            node_feats = layer(g_list[1].to(device), node_feats).to(device)

        out_put_1 = F.adaptive_max_pool1d(node_feats.unsqueeze(0).permute(0, 2, 1), 1).squeeze()

        x11 = out_put_1 * 1/self.num_net

        node_feats = g_list[0].ndata['feat']
        for layer in self.drug1_gcn2:
            node_feats = layer(g_list[0].to(device), node_feats).to(device)

        out_put_2 = F.adaptive_max_pool1d(node_feats.unsqueeze(0).permute(0, 2, 1), 1).squeeze()

        x11 = x11 + out_put_2 * 1/self.num_net
        xc = self.fc1(x11)
        xc = self.sigmoid(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out
    


    



class GATlayer(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads):
        super(GATlayer, self).__init__()
        
        # 初始化 GATConv，不使用权重参数
        self.conv1 = dglnn.GATConv(in_feats, hidden_feats, num_heads=num_heads)
        self.conv2 = dglnn.GATConv(hidden_feats * num_heads, out_feats, num_heads=1)
        
    def initialize_weights(self, layer, edge_feats):
        if hasattr(layer, 'weight'):
            # 使用边的特征初始化权重
            layer.weight.data = edge_feats.view(-1, 1, 1).to(layer.weight.dtype)

    def forward(self, g, features, edge_feats):
        # 手动初始化第一层 GAT 的权重
        self.initialize_weights(self.conv1, edge_feats)

        # 第一层 GAT
        x = self.conv1(g, features).flatten(1)
        x = F.relu(x)

        # 第二层 GAT
        x = self.conv2(g, x).flatten(1)
        return x




class GAT(torch.nn.Module):
    def __init__(self, in_feats = 64, output_dim=64, dropout=0.2, out_feats=16):
        super(GAT, self).__init__()
        
        self.node_rbf_layer_1_path = RBFKernelMappingLayer(input_dim=2, output_dim=output_dim, gamma=0.1)
        self.node_rbf_layer = RBFKernelMappingLayer(input_dim=1, output_dim=output_dim, gamma=0.1)


        self.node_rbf_layer = RBFKernelMappingLayer(input_dim=62, output_dim=output_dim, gamma=0.1)
        self.edge_rbf_layer = RBFKernelMappingLayer(input_dim=2, output_dim=output_dim, gamma=0.1)

        self.gat_1 = GATlayer(in_feats=in_feats, hidden_feats=8, out_feats=1, num_heads=2)

        self.gat_2 = GATlayer(in_feats=in_feats, hidden_feats=8, out_feats=output_dim, num_heads=2)

        # combined layers
        self.fc1 = nn.Linear(output_dim, out_feats)
        self.out = nn.Linear(out_feats, 1)
        # activation and regularization
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.LeakyReLU = nn.LeakyReLU()


    def forward(self, g_list,device):
        node_feats = g_list[1].ndata['feat']
        edge_feats = g_list[1].edata['feat']
        node_feats = (torch.tensor(self.node_rbf_layer_1_path(node_feats))).to(device)

        out_put_1 = self.gat_1(g_list[1].to(device), node_feats, edge_feats)

        out_1 = out_put_1
        
        m, n = out_put_1.shape

        # 使用 torch.cat 生成 2m x n 张量 B
        B = torch.cat([out_put_1[i:i+1].repeat(2, 1) for i in range(m)], dim=0)
        out_put_1 = B.to(device)

        node_feats = g_list[0].ndata['feat']
        edge_feats = g_list[0].edata['feat'].to(device)

        node_feats = (torch.tensor(self.node_rbf_layer(node_feats))).to(device)

        concatenated_tensor = torch.cat((out_put_1.unsqueeze(0), edge_feats.unsqueeze(0)), dim=0).to(device)
        edge_feats = torch.add(concatenated_tensor[0], concatenated_tensor[1]).to(device)

        out_put_2 = self.gat_2(g_list[0].to(device), node_feats, edge_feats)
        
        out_2 = out_put_2
        out_2 = torch.mean(out_put_2, dim=0, keepdim=True)
        
        #y = torch.cat((out_2, out_1), dim=0).to(device)
        h = self.relu(out_2).to(device)
        h = self.dropout(h).to(device)
        h = self.fc1(h).to(device)
        
        h = self.relu(h).to(device)
        h = self.out(h).to(device)
    
        #print(h.size)
        out = torch.squeeze(h, dim=1)
        return out





#构建一个GraphSAGE模型
#构建一个GraphSAGE模型
class SAGE(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout=0.2, num_layers=3):
        super().__init__()

        self.in_feats = in_feats
        self.hidden_size = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.graphsage_1_layers = nn.ModuleList()
        self.graphsage_2_layers = nn.ModuleList()

        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        #linear_graph
        self.node_rbf_layer_1_path = RBFKernelMappingLayer(input_dim=in_feats[1], output_dim=hid_feats, gamma=0.1)
        self.layer2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean')
        self.graphsage_1_layers.append(self.layer2 )
        for _ in range(num_layers - 1):
            self.graphsage_1_layers.append(dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean'))
        
        #graph
        self.node_rbf_layer = RBFKernelMappingLayer(input_dim=in_feats[0], output_dim=hid_feats, gamma=0.1)
        self.layer3 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean')
        self.graphsage_2_layers.append(self.layer3 )
        for _ in range(num_layers - 1):
            self.graphsage_2_layers.append(dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean'))

        # combined layers
        self.fc1 = nn.Linear(hid_feats*2, hid_feats)
        self.out = nn.Linear(hid_feats, out_feats)

        # activation and regularization
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.LeakyReLU = nn.LeakyReLU()

        #output layer
        self.read_out = nn.Linear(out_feats,1)

    def forward(self, graphs, node_feats, edge_feats=None):

        # 输入是节点的特征
        g0, g1 = graphs
        node_feats = g1.ndata['feat'].to(device)
        node_feats = self.node_rbf_layer_1_path(node_feats).to(device)
        for layer in self.graphsage_1_layers:
            node_feats = layer(g1.to(device), node_feats).to(device)
        y1 = node_feats.to(device)

        node_feats = g0.ndata['feat'].to(device)
        node_feats = self.node_rbf_layer(node_feats).to(device)

        for layer in self.graphsage_2_layers:
            node_feats = layer(g0.to(device), node_feats).to(device)
        y2 = node_feats.to(device)

        y2_pooled = F.adaptive_avg_pool1d(y2.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        y1_pooled = F.adaptive_avg_pool1d(y1.unsqueeze(0).permute(0, 2, 1), 1).squeeze()

        # 拼接池化后的结果
        y = torch.cat((y2_pooled, y1_pooled), dim=0)
        
        h = self.fc1(y).to(device)
        h = self.relu(h).to(device)
        h = self.dropout(h).to(device)

        h = self.out(h).to(device)
        h = self.LeakyReLU(h).to(device)
        out = self.read_out(h)

        return out



    

class QCBlock(MessagePassing):
    def __init__(self, in_size, out_size, device):
        super(QCBlock, self).__init__(aggr='add')
        self.in_size = in_size
        self.out_size = out_size
        self.K_v2v = nn.Parameter(torch.zeros(in_size, out_size, device=device))
        torch.nn.init.xavier_uniform_(self.K_v2v)
        self.K_e2v = nn.Parameter(torch.zeros(in_size, out_size, device=device))
        torch.nn.init.xavier_uniform_(self.K_e2v)
        self.V_v2v = nn.Parameter(torch.zeros(in_size, out_size, device=device))
        torch.nn.init.xavier_uniform_(self.V_v2v)
        self.V_e2v = nn.Parameter(torch.zeros(in_size, out_size, device=device))
        torch.nn.init.xavier_uniform_(self.V_e2v)

        self.linear_update = nn.Linear(out_size * 2, out_size * 2, device=device)
        self.layernorm = nn.LayerNorm(out_size * 2, device=device)
        self.sigmoid = nn.Sigmoid()
        self.msg_layer = nn.Sequential(nn.Linear(out_size * 2, out_size, device=device),nn.LayerNorm(out_size, device=device))

    def forward(self, x, edge_index, edge_feature):
        K_v = torch.mm(x, self.K_v2v)
        V_v = torch.mm(x, self.V_v2v)

        if min(edge_feature.shape) == 0:
            return V_v
        else:
            out = self.propagate(edge_index, query_v=K_v, key_v=K_v, value_v=V_v, edge_feature=edge_feature)
            return out

    def message(self, query_v_i, key_v_i, key_v_j, edge_feature, value_v_i, value_v_j):
        K_E = torch.mm(edge_feature, self.K_e2v)
        V_E = torch.mm(edge_feature, self.V_e2v)

        query_i = torch.cat([query_v_i, query_v_i], dim=1)
        key_j = torch.cat([key_v_j, K_E], dim=1)
        alpha = (query_i * key_j) / math.sqrt(self.out_size * 2)
        alpha = F.dropout(alpha, p=0, training=self.training)
        out = torch.cat([value_v_j, V_E], dim=1)
        out = self.linear_update(out) * self.sigmoid(self.layernorm(alpha.view(-1, 2 * self.out_size)))
        out = torch.nn.functional.leaky_relu(self.msg_layer(out))
        return out

class QCConv(nn.Module):
    def __init__(self, in_size, out_size, head, device):
        super(QCConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.head = head
        self.attention = nn.ModuleList()
        for _ in range(self.head):
            self.attention.append(QCBlock(in_size, out_size, device))
        self.linear_concate_v = nn.Linear(out_size * head, out_size, device=device)
        self.linear_concate_e = nn.Linear(out_size * head, out_size, device=device)
        self.linear_input = nn.Linear(in_size,out_size,device=device)
        self.bn_v = nn.BatchNorm1d(out_size, device=device)
        self.bn_e = nn.BatchNorm1d(out_size, device=device)
        self.bn = nn.BatchNorm1d(out_size, device=device)

        self.coe1 = nn.Parameter(torch.tensor([0.5], device=device))
        self.coe2 = nn.Parameter(torch.tensor([0.5], device=device))

        self.final = nn.Linear(in_size, out_size, device=device)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear_concate_v.reset_parameters()
        self.linear_concate_e.reset_parameters()
        self.final.reset_parameters()


    def forward(self, x, edge_index, edge_feature):
        hidden_v = []
        for atten in self.attention:
            hv = atten(x, edge_index, edge_feature)
            hidden_v.append(hv)

        hv = torch.cat(hidden_v, dim=1)
        out = self.linear_concate_v(hv)
        out = F.leaky_relu(self.bn_v(out))
        x = self.linear_input(x)
        out = out + x

        return out

class QCformer(nn.Module):
    def __init__(self, in_size, out_size, edge_size, head1, head2, layer_number, device):
        super(QCformer, self).__init__()

        self.layer_number = layer_number
        self.in_size = in_size
        self.out_size = out_size
        self.edge_size = edge_size
        self.head1 = head1
        self.head2 = head2

        # RBF核函数的参数
        self.rbf_layer1 = RBFKernelMappingLayer(input_dim=edge_size, output_dim=in_size, gamma=0.1)
        self.node_rbf_layer1 = RBFKernelMappingLayer(input_dim=1, output_dim=in_size, gamma=0.1)

        self.rbf_layer2 = RBFKernelMappingLayer(input_dim=edge_size, output_dim=in_size, gamma=0.1)
        self.node_rbf_layer2 = RBFKernelMappingLayer(input_dim=5, output_dim=in_size, gamma=0.1)
        #self.edge_layer = nn.Linear(in_size, out_size, device=device)

        self.layer1 = nn.ModuleList([QCConv(in_size, in_size, head1, device) for _ in range(self.layer_number)])

        self.layer2 = nn.ModuleList([QCConv(in_size, in_size, head1, device) for _ in range(self.layer_number)])


        self.fc = nn.Sequential(
            nn.Linear(in_size * 2, out_size, device=device), nn.LeakyReLU()
        )
        self.fc_out = nn.Sequential(
            nn.Linear(out_size, 1, device=device), nn.LeakyReLU()
        )


    def forward(self, g_list, node_feats, edge_feats=None):
        g0, g1 = g_list
        node_feature = g1.ndata['feat'].to(device)
        edge_feature = g1.edata['feat'].to(device)
        g1 = g1.to(device)
        g0 = g0.to(device)

        edges = g1.edges()
        #print(edge_feature.shape)
        edge_index = torch.tensor([list(edges[0]), list(edges[1])], dtype=torch.long).to(device)

        #print(edge_feature.shape)
        edge_feature = self.rbf_layer1(edge_feature).to(device)
        
        node_feature = self.node_rbf_layer1(node_feature).to(device)
        for _ in range(self.layer_number):
            x = self.layer1[_](node_feature, edge_index, edge_feature).to(device)

        #print(x.shape)
        #print(edge_feature.shape)
        #feature1 = global_mean_pool(x, g1.nodes()).to(device)
        feature1 = F.adaptive_avg_pool1d(x.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        #print(feature1.shape)
        edges = g0.edges()
        edge_index = torch.tensor([list(edges[0]), list(edges[1])], dtype=torch.long).to(device)

        node_feature = g0.ndata['feat'].to(device)
        edge_feature = g0.edata['feat'].to(device)
        
        #print(edge_feature.shape)
        edge_feature = edge_feature.unsqueeze(1).unsqueeze(2)
        edge_feature = self.rbf_layer2(edge_feature).to(device)
        
        node_feature = self.node_rbf_layer2(node_feature).to(device)

        #print(x.shape)
        #print(edge_feature.shape)
        concatenated_tensor = torch.cat((x.unsqueeze(0), edge_feature.unsqueeze(0)), dim=0).to(device)
        edge_feature = torch.add(concatenated_tensor[0], concatenated_tensor[1]).to(device)

        for _ in range(self.layer_number):
            y = self.layer2[_](node_feature, edge_index, edge_feature).to(device)


        #feature2 = global_mean_pool(y, g0.nodes()).to(device)
        feature2 = F.adaptive_avg_pool1d(y.unsqueeze(0).permute(0, 2, 1), 1).squeeze()

        feature = torch.cat([feature1, feature2], dim=0).to(device)

        feature = self.fc(feature)
        out = self.fc_out(feature)

        return out
