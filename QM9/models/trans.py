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
from models.edge_node import edge_node_block
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

        pooled_outputs.append(pooled_out)

    return torch.stack(pooled_outputs)



class QKV_Block(MessagePassing):
    def __init__(self,in_size,out_size):
        super(QKV_Block,self).__init__(aggr='add')
        self.in_size = in_size
        self.out_size = out_size
        self.K_v2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.K_v2v)
        self.K_e2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.K_e2v)
        self.V_v2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.V_v2v)
        self.V_e2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.V_e2v)
        
        self.linear_update = torch.nn.Linear(out_size*2,out_size*2,device=device)
        self.layernorm = torch.nn.LayerNorm(out_size*2,device=device)
        self.sigmoid = torch.nn.Sigmoid()
        self.msg_layer = torch.nn.Sequential(torch.nn.Linear(out_size*2,out_size,device=device),torch.nn.LayerNorm(out_size,device=device) )
        
        
        
    def forward(self,x,edge_index,edge_feature):
        K_v = torch.mm(x,self.K_v2v)
        V_v = torch.mm(x,self.V_v2v)
        
        
        if min(edge_feature.shape)==0:
            return V_v
        else:
            out = self.propagate(edge_index,query_v=K_v,key_v=K_v,value_v=V_v,edge_feature=edge_feature)
            return out
        
    
    def message(self,query_v_i,key_v_i,key_v_j,edge_feature,value_v_i,value_v_j):
        K_E = torch.mm(edge_feature,self.K_e2v)
        V_E = torch.mm(edge_feature,self.V_e2v)
        
        query_i = torch.cat([ query_v_i,query_v_i  ],dim=1)
        key_j = torch.cat([ key_v_j,K_E ],dim=1)
        alpha = ( query_i * key_j ) / math.sqrt(self.out_size * 2)
        alpha = F.dropout(alpha,p=0,training=self.training)
        out = torch.cat([ value_v_j,V_E  ],dim=1)
        out = self.linear_update(out) * self.sigmoid( self.layernorm(alpha.view(-1,2*self.out_size)) )
        out = torch.nn.functional.leaky_relu( self.msg_layer(out) )
        return out
        
    




class Trans_Conv(nn.Module):
    def __init__(self, in_size, out_size, head):
        super(Trans_Conv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.head = head
        self.attention = nn.ModuleList()
        for _ in range(self.head):
            self.attention.append(QKV_Block(in_size, out_size))
        self.linear_concate_v = nn.Linear(out_size * head, out_size)
        self.linear_concate_e = nn.Linear(out_size * head, out_size)
        self.linear_input = nn.Linear(in_size,out_size)
        self.bn_v = nn.BatchNorm1d(out_size)
        self.bn_e = torch.nn.BatchNorm1d(out_size,device=device)
        self.bn = torch.nn.BatchNorm1d(out_size,device=device)
        
        self.coe1 = torch.nn.Parameter(torch.tensor([0.5],device=device))
        self.coe2 = torch.nn.Parameter(torch.tensor([0.5],device=device))
        self.final = torch.nn.Linear(in_size,out_size,device=device)
        self.reset_parameters()


    def reset_parameters(self):
        self.linear_concate_v.reset_parameters()
        self.linear_concate_e.reset_parameters()
        self.final.reset_parameters()


    def forward(self,x,edge_index,edge_feature):
        
        hidden_v = []
        for atten in self.attention:
            hv = atten(x,edge_index,edge_feature)
            hidden_v.append(hv)
            
        hv = torch.cat(hidden_v,dim=1)
        out = self.linear_concate_v(hv)
        out = torch.nn.functional.leaky_relu( self.bn_v(out))
        out = out + x
        return out



class Transformer_Block(nn.Module):
    def __init__(self, in_size, out_size, out_dim, edge_size, hidden_size, head, layer_number):
        super(Transformer_Block, self).__init__()

        '''
        >>> Edge to Node
        in_size = input_node_dim
        out_size = out_node_dim
        edge_size = input_edge_dim
        >>> Transformer node
        out_size = input_node_dim
        out_dim  = output_node_dim
        '''
        self.layer_number = layer_number
        self.in_size = in_size
        self.out_size = out_size
        self.edge_size = edge_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.head = head
 

        # RBF核函数的参数
        #self.Edge_Node_layer = edge_node_block(in_size, hidden_size, out_size, edge_size)
        self.Edge_Node_layer = nn.Linear(self.edge_size, self.in_size)
        self.Trans_layer = nn.ModuleList([Trans_Conv(out_size, out_dim, head) for _ in range(self.layer_number)])


    def forward(self, graph):

        node_feat = graph.ndata['feat'].to(device)
        edge_feat = graph.edata['feat'].to(device)
        edge_feat = self.Edge_Node_layer(edge_feat)
        #node_feats = self.Edge_Node_layer(graph, node_feat, edge_feat).to(device)

        edges = graph.edges()
        edge_index = torch.tensor([list(edges[0]), list(edges[1])], dtype=torch.long).to(device)

        for _ in range(self.layer_number):
            node_feat = self.Trans_layer[_](node_feat, edge_index,edge_feat)

        return node_feat


class Transformer(nn.Module):
    def __init__(self, encode_dim, out_size, hidden_size, head, layer_number):
        super(Transformer, self).__init__()

        self.encode_dim = encode_dim
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.angle_feature_size = encode_dim[2]
        self.dihedral_feature_size = encode_dim[3]
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.head = head
        self.layer_number = layer_number

        self.fg_node_transformer = Transformer_Block(in_size=self.angle_feature_size, out_size=self.angle_feature_size, out_dim = self.angle_feature_size , edge_size =self.dihedral_feature_size, hidden_size = self.hidden_size, head = self.head, layer_number = self.layer_number).to(device)

        self.lg_node_transformer = Transformer_Block(in_size=self.bond_feature_size, out_size=self.bond_feature_size, out_dim = self.bond_feature_size , edge_size =self.angle_feature_size, hidden_size = self.hidden_size, head = self.head, layer_number = self.layer_number).to(device)

        self.g_node_transformer = Transformer_Block(in_size=self.atom_feature_size, out_size=self.atom_feature_size, out_dim = self.atom_feature_size , edge_size =self.bond_feature_size, hidden_size = self.hidden_size, head = self.head, layer_number = self.layer_number).to(device)

        

        self.leaky_relu = nn.LeakyReLU()

        layers3 = [
            nn.Linear(sum(self.encode_dim[0:3]), self.hidden_size),
            self.leaky_relu,
            #self.activation,
            nn.Linear(self.hidden_size, self.out_size),
            nn.Sigmoid()
        ]
        # Create a sequential model
        self.Readout3 = nn.Sequential(*layers3)

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


        g_graph = g.to(device) 
        g_graph_edge_feat = g_graph.edata['feat'].to(device)
        #ini_y1 = self.activation(g_graph_node_feat)


        lg_graph = lg.to(device) 
        lg_graph_edge_feat = lg_graph.edata['feat'].to(device)
        #ini_y2 = self.activation(lg_graph_node_feat)
        #ini_y2 = lg_graph_node_feat


        fg_graph = fg.to(device) 
        fg_graph_node_feat = fg_graph.ndata['feat'].to(device)
        #fg_graph_edge_feat = fg_graph.edata['feat'].to(device)
        #fg_graph_edge_feat = self.fg_edge_node(fg_graph_edge_feat)

        fg_graph_node_feat = self.fg_node_transformer(fg_graph)
        angle_feats = torch.add(lg_graph_edge_feat,fg_graph_node_feat) 
        lg_graph.edata['feat'] = angle_feats

        lg_graph_node_feat = self.lg_node_transformer(lg_graph)
        bond_feats = torch.add(g_graph_edge_feat,lg_graph_node_feat) 
        g_graph.edata['feat'] = bond_feats

        g_graph_node_feat = self.g_node_transformer(g_graph)

        
        out_1 = g_graph_node_feat.to(device)
        out_2 = lg_graph_node_feat.to(device)
        out_3 = fg_graph_node_feat.to(device)

        if pooling == 'avg':
            #y1 = AvgPooling(out_1, g)
            #y2 = AvgPooling(out_2, lg)
            #y3 = AvgPooling(out_3, fg)

            y1 = pool_subgraphs_node(out_1, g)
            y2 = pool_subgraphs_node(out_2, lg)
            y3 = pool_subgraphs_node(out_3, fg)
            #y4 = pool_subgraphs_edge(out_4, fg)       

        y = torch.cat((y1, y2, y3), dim=1)
        batch_out = y.squeeze() 

        out = self.Readout3(batch_out)

        return out