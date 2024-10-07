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
from sklearn.kernel_approximation import RBFSampler
from dgl.nn import GlobalAttentionPooling, SortPooling
from dgl.utils import expand_as_pair 
from dgl.nn.functional import edge_softmax
from dgl import DGLError
from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling





device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")



def swish(x):
    """
    Swish activation function,
    from Ramachandran, Zopf, Le 2017. "Searching for Activation Functions"
    """
    return x * torch.sigmoid(x)




 
class Path_Complex_LIH_layer(nn.Module):
    def __init__(self, in_l_dim, in_m_dim, in_x_dim, out_l_dim, out_m_dim, out_x_dim, num_heads, bias=True):
        super(Path_Complex_LIH_layer, self).__init__()
        '''
        in_l_dim: input n-1 path dim
        in_m_dim: input n path dim
        in_x_dim: input n+1 path dim

        out_l_dim: output n-1 path dim
        out_m_dim: output n path dim
        out_x_dim: output n+1 path dim
        '''
        self._num_heads = num_heads
        self._out_l_dim = out_l_dim
        self._out_m_dim = out_m_dim
        self._out_x_dim = out_x_dim 

        self.lg_fc_node = nn.Linear(in_m_dim+in_x_dim, out_m_dim * num_heads, bias=True)
        self.lg_fc_ni = nn.Linear(in_m_dim, out_x_dim * num_heads, bias=False)
        self.lg_fc_fij = nn.Linear(in_x_dim, out_x_dim * num_heads, bias=False)
        self.lg_fc_nj = nn.Linear(in_m_dim, out_x_dim * num_heads, bias=False)
        self.lg_attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_x_dim)).to(device))

        self.g_fc_node = nn.Linear(in_m_dim+in_l_dim, out_m_dim * num_heads, bias=True)
        self.g_fc_ni = nn.Linear(in_l_dim, out_m_dim * num_heads, bias=False)
        self.g_fc_fij = nn.Linear(in_m_dim, out_m_dim * num_heads, bias=False)
        self.g_fc_nj = nn.Linear(in_l_dim, out_m_dim * num_heads, bias=False)
        self.g_attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_m_dim)).to(device))


        self.m_linear = nn.Linear(out_m_dim*2, out_m_dim, bias=True)

        if bias:
            self.bias_lg = nn.Parameter(torch.FloatTensor(size=(num_heads * out_x_dim,)))
            self.bias_g = nn.Parameter(torch.FloatTensor(size=(num_heads * out_m_dim,)))
        else:
            self.register_buffer('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.lg_fc_node.weight)
        nn.init.xavier_normal_(self.lg_fc_ni.weight)
        nn.init.xavier_normal_(self.lg_fc_fij.weight)
        nn.init.xavier_normal_(self.lg_fc_nj.weight)
        nn.init.xavier_normal_(self.lg_attn)

        nn.init.xavier_normal_(self.g_fc_node.weight)
        nn.init.xavier_normal_(self.g_fc_ni.weight)
        nn.init.xavier_normal_(self.g_fc_fij.weight)
        nn.init.xavier_normal_(self.g_fc_nj.weight)
        nn.init.xavier_normal_(self.g_attn)
        
        nn.init.xavier_normal_(self.m_linear.weight)


        if self.bias_g is not None:
            nn.init.constant_(self.bias_g, 0)
        if self.bias_lg is not None:
            nn.init.constant_(self.bias_lg, 0)

    # 定义消息传递函数，将边的特征发送给相应的源节点和目标节点
    def message_func(self, edges):
        return {'feat': edges.data['feat']}


    def reduce_func(self, nodes):
        num_edges = nodes.mailbox['feat'].size(1)  # 消息数量
        if num_edges > 0:
            agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges  # 求平均
        else:
            agg_feats = torch.zeros((nodes.data['feat'].size(0), nodes.data['feat'].size(1)))  # 对于孤立点，使用零向量
        return {'agg_feats': agg_feats}
      

    def forward(self, graph, l_graph, l_feats, m_feats, x_feats, get_attention=False):
        with graph.local_scope():

            #graph.ndata['feat'] = l_feats
            #graph.edata['feat'] = m_feats

            #l_graph.ndata['feat'] = m_feats
            #l_graph.edata['feat'] = x_feats

            in_degrees = graph.in_degrees().float().unsqueeze(-1)
            in_degrees[in_degrees == 0] = 1  # 将入度为0的节点设置为1，以避免除零错误
            lg_f_ni = self.lg_fc_ni(m_feats)
            lg_f_nj = self.lg_fc_nj(m_feats)
            lg_f_fij = self.lg_fc_fij(x_feats)
            l_graph.srcdata.update({'f_ni': lg_f_ni})
            l_graph.dstdata.update({'f_nj': lg_f_nj})
            # add ni, nj factors
            l_graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            # add fij to node factor
            lg_f_out = l_graph.edata.pop('f_tmp') + lg_f_fij
            if self.bias_lg is not None:
                lg_f_out = lg_f_out + self.bias_lg
            lg_f_out = nn.functional.leaky_relu(lg_f_out)
            x_feats = lg_f_out.view(-1, self._num_heads, self._out_x_dim)
            # compute attention factor
            lg_e = (x_feats * self.lg_attn).sum(dim=-1).unsqueeze(-1)
            l_graph.edata['a'] = edge_softmax(l_graph, lg_e)

            # 发送消息并接收消息，将边的特征分配给边的两个端点
            l_graph.send_and_recv(l_graph.edges(), self.message_func, reduce_func=self.reduce_func)
            m_feats_1 = torch.cat((l_graph.ndata['feat'],l_graph.ndata['agg_feats']),dim=1)

            l_graph.ndata['h_out'] = self.lg_fc_node(m_feats_1).view(-1, self._num_heads,
                                                             self._out_m_dim)
            # calc weighted sum
            l_graph.update_all(fn.u_mul_e('h_out', 'a', 'm'),
                             fn.sum('m', 'h_out'))
            
            lg_h_out = nn.functional.leaky_relu(l_graph.ndata['h_out'])
            lg_h_out = lg_h_out.view(-1, self._num_heads, self._out_m_dim)

            lg_h_out = torch.sum(lg_h_out, dim=1)
            lg_f_out = torch.sum(lg_f_out, dim=1)



            g_f_ni = self.g_fc_ni(l_feats)
            g_f_nj = self.g_fc_nj(l_feats)
            g_f_fij = self.g_fc_fij(graph.edata['feat'])
            graph.srcdata.update({'f_ni': g_f_ni})
            graph.dstdata.update({'f_nj': g_f_nj})
            # add ni, nj factors
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            g_f_out = graph.edata.pop('f_tmp') + g_f_fij
            if self.bias_g is not None:
                g_f_out = g_f_out + self.bias_g
            g_f_out = nn.functional.leaky_relu(g_f_out)
            #g_f_out = nn.functional.silu(g_f_out)
            g_f_out = g_f_out.view(-1, self._num_heads, self._out_m_dim)
            # compute attention factor
            g_e = (g_f_out * self.g_attn).sum(dim=-1).unsqueeze(-1)
            # add fij to node factor
            
            src,dst = graph.edges()
            srcdata_edge = graph.ndata['feat'][src]
            dstdata_edge = graph.ndata['feat'][dst]
            node_edeg = torch.add(srcdata_edge,dstdata_edge)
            m_feats_2 = torch.cat((node_edeg,graph.edata['feat']),dim =1)
        
            l_graph.ndata['h_out'] = self.g_fc_node(m_feats_2).view(-1, self._num_heads,self._out_m_dim)

            l_graph.ndata['a'] = edge_softmax(graph, g_e)

            # calc weighted sum
            l_graph.update_all(fn.u_mul_v('h_out', 'a', 'm'),
                             fn.sum('m', 'h_out'))

            g_h_out = nn.functional.leaky_relu( l_graph.ndata['h_out'])
            g_h_out = g_h_out.view(-1, self._num_heads, self._out_m_dim)
            g_h_out = torch.sum(g_h_out, dim=1)
            g_f_out = torch.sum(g_f_out, dim=1)

            m_feats = torch.add(g_h_out, lg_h_out)

            l_graph.ndata['feat'] = m_feats
            return m_feats





class Path_Complex_IH_layer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads, bias=True):
        super(Path_Complex_IH_layer, self).__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_node = nn.Linear(in_node_feats+in_edge_feats, out_node_feats * num_heads, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats * num_heads, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)).to(device))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_edge_feats,)).to(device))
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

    def message_func(self, edges):
        return {'feat': edges.data['feat']}

    def reduce_func(self, nodes):
        # 归一化或平均
        num_edges = nodes.mailbox['feat'].size(1)  # 计算接收到的消息的数量
        agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges  # 求平均
        return {'agg_feats': agg_feats}
    

    def forward(self, graph, nfeats, efeats, get_attention=False):
        with graph.local_scope():
            graph.ndata['feat'] = nfeats
            graph.edata['feat'] = efeats
            in_degrees = graph.in_degrees().float().unsqueeze(-1)
            in_degrees[in_degrees == 0] = 1  # 将入度为0的节点设置为1，以避免除零错误
            f_ni = self.fc_ni(nfeats)# in_node_feats --> out_edge_feats
            f_nj = self.fc_nj(nfeats)# in_node_feats --> out_edge_feats
            f_fij = self.fc_fij(efeats)# in_edge_feats --> out_edge_feats

            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            
            f_out = graph.edata.pop('f_tmp') + f_fij
            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = nn.functional.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
            
            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)

            graph.send_and_recv(graph.edges(), self.message_func, reduce_func=self.reduce_func)
            m_feats = torch.cat((graph.ndata['feat'],graph.ndata['agg_feats']),dim=1)
            
            graph.edata['a'] = edge_softmax(graph, e)
            graph.ndata['h_out'] = self.fc_node(m_feats).view(-1, self._num_heads, self._out_node_feats)
            
            graph.update_all(fn.u_mul_e('h_out', 'a', 'm'),
                             fn.sum('m', 'h_out'))

            h_out = nn.functional.leaky_relu(graph.ndata['h_out'])
            h_out = h_out.view(-1, self._num_heads, self._out_node_feats)

            h_out = torch.sum(h_out, dim=1)
            f_out = torch.sum(f_out, dim=1)
            
            if get_attention:
                return h_out, f_out, graph.edata.pop('a')
            else:
                return h_out, f_out



'''     

class HL_LH_block_1(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, num_heads, num_layers):
        super(HL_LH_block_1, self).__init__()
        atom_f, bond_f, angle_f, dihedral_f = encode_dim

        # Define layers using the updated create_layers method
        self.block_0_s = self.create_layers(Path_Complex_layer, bond_f, angle_f, dihedral_f, bond_f, angle_f, dihedral_f,num_heads, num_layers=num_layers-1)

        self.block_1_s = self.create_layers(Path_Complex_layer, atom_f, bond_f, angle_f, atom_f, bond_f, angle_f, num_heads,num_layers=num_layers-1)

        self.block_2_s = self.create_layers(EGATlayer, atom_f, bond_f, atom_f, bond_f,num_heads, num_layers=num_layers)

        self.block_3_s = self.create_layers(Path_Complex_layer, atom_f, bond_f, angle_f, atom_f, bond_f, angle_f, num_heads,num_layers=num_layers-1)

        self.block_4_s = self.create_layers(Path_Complex_layer, bond_f, angle_f, dihedral_f, bond_f, angle_f, dihedral_f, num_heads,num_layers=num_layers-1)

    def create_layers(self, layer_class, *args, num_layers):
        return nn.ModuleList([layer_class(*args, ) for _ in range(num_layers)])

    def apply_layers(self, layers, *inputs):
        for layer in layers:
            inputs = layer(*inputs)
        return inputs

    def forward(self, g, lg, fg, device, g_node_feat, g_edge_feat, lg_node_feat, lg_edge_feat, fg_node_feat, fg_edge_feat):
        x_feat = fg_node_feat
        m_feat = lg_node_feat
        l_feat = g_node_feat

        x_feat = self.apply_layers(self.block_0_s, lg, fg, lg_node_feat, fg_node_feat, fg_edge_feat, x_feat)
        m_feat = self.apply_layers(self.block_1_s, g, lg, g_node_feat, lg_node_feat, x_feat)
        l_feat, g_edge_feat = self.apply_layers(self.block_2_s, g, l_feat, g_edge_feat)
        m_feat = self.apply_layers(self.block_3_s, g, lg, g_node_feat, m_feat, lg_edge_feat)
        x_feat = self.apply_layers(self.block_4_s, lg, fg, lg_node_feat, x_feat, fg_edge_feat)

        return F.leaky_relu(g_node_feat + l_feat), F.leaky_relu(lg_node_feat + m_feat), F.leaky_relu(fg_node_feat + x_feat)


'''
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

        
        
        self.block_0_s = nn.ModuleList()
        self.block_1_s = nn.ModuleList()
        self.block_2_s = nn.ModuleList()
        self.block_3_s = nn.ModuleList()
        self.block_4_s = nn.ModuleList()

        self.block_0 = Path_Complex_LIH_layer(in_l_dim=self.bond_feature_size,in_m_dim=self.angle_feature_size,in_x_dim=self.dihedral_feature_size,out_l_dim=self.bond_feature_size,out_m_dim=self.angle_feature_size,out_x_dim=self.dihedral_feature_size,num_heads=self.num_heads)

        for _ in range(self.num_layers-1):
            self.block_0_s.append(self.block_0)
            
        

        self.block_1 = Path_Complex_LIH_layer(in_l_dim=self.atom_feature_size,in_m_dim=self.bond_feature_size,in_x_dim=self.angle_feature_size,out_l_dim=self.atom_feature_size,out_m_dim=self.bond_feature_size,out_x_dim=self.angle_feature_size,num_heads=self.num_heads)

        for _ in range(self.num_layers-1):
            self.block_1_s.append(self.block_1)
            
        
        
        self.block_2 = Path_Complex_IH_layer(in_node_feats=self.atom_feature_size,in_edge_feats=self.bond_feature_size,out_node_feats=self.atom_feature_size,out_edge_feats=self.bond_feature_size,num_heads=self.num_heads)
        for _ in range(self.num_layers):
            self.block_2_s.append(self.block_2)
        


        self.block_3 = Path_Complex_LIH_layer(in_l_dim=self.atom_feature_size,in_m_dim=self.bond_feature_size,in_x_dim=self.angle_feature_size,out_l_dim=self.atom_feature_size,out_m_dim=self.bond_feature_size,out_x_dim=self.angle_feature_size,num_heads=self.num_heads)

        for _ in range(self.num_layers-1):
            self.block_3_s.append(self.block_3)


        self.block_4 = Path_Complex_LIH_layer(in_l_dim=self.bond_feature_size,in_m_dim=self.angle_feature_size,in_x_dim=self.dihedral_feature_size,out_l_dim=self.bond_feature_size,out_m_dim=self.angle_feature_size,out_x_dim=self.dihedral_feature_size,num_heads=self.num_heads)

        for _ in range(self.num_layers-1):
            self.block_4_s.append(self.block_4)
        
        
        
        
        

    def forward(self, g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat):

        
        ### main PCNN
        if len(self.block_0_s) > 0:
            for layer in self.block_0_s:
                x_feat = layer(lg, fg, lg_node_feat,fg_node_feat,fg_edge_feat)
            x_feat = torch.add(fg_node_feat, x_feat)

        if len(self.block_1_s) > 0:
            for layer in self.block_1_s:
                m_feat = layer(g, lg, g_node_feat,lg_node_feat, x_feat)
            m_feat = torch.add(lg_node_feat, m_feat)

        
        if len(self.block_2_s) > 0:
            for i in range(len(self.block_2_s)):
                layer = self.block_2_s[i]
                if i == 0:
                    l_feat, g_edge_feat = layer(g, g_node_feat, m_feat)
                #elif i == 1:
                else:
                    l_feat, g_edge_feat = layer(g, l_feat, g_edge_feat)
            l_feat = torch.add(g_node_feat, l_feat)

        l_feat = nn.functional.leaky_relu(torch.add(g_node_feat, l_feat))

        if len(self.block_3_s) > 0:
            for layer in self.block_3_s:
                m_feat = layer(g, lg, g_node_feat, m_feat, lg_edge_feat)
            m_feat = torch.add(lg_node_feat, m_feat)
        m_feat = nn.functional.leaky_relu(torch.add(lg_node_feat, m_feat))
        
        if len(self.block_4_s) > 0:
            for layer in self.block_4_s:
                x_feat = layer(lg, fg, lg_node_feat, x_feat, fg_edge_feat)
            #x_feat = torch.add(fg_node_feat, x_feat)

        x_feat = nn.functional.leaky_relu(torch.add(fg_node_feat, x_feat))

        return l_feat, m_feat, x_feat
        


        '''
        ### updata 0,1
        if len(self.block_2_s) > 0:
            for i in range(len(self.block_2_s)):
                layer = self.block_2_s[i]
                if i == 0:
                    l_feat, g_edge_feat = layer(g, g_node_feat, g_edge_feat)
                if i == 1:
                    l_feat, g_edge_feat = layer(g, l_feat, g_edge_feat)
            l_feat = torch.add(g_node_feat, l_feat)

        l_feat = nn.functional.leaky_relu(torch.add(g_node_feat, l_feat))

        
        return l_feat, lg_node_feat, fg_node_feat
        '''


        '''
        ### 0,1,2
        if len(self.block_1_s) > 0:
            for layer in self.block_1_s:
                m_feat = layer(g, lg, g_node_feat,lg_node_feat, fg_node_feat)
            m_feat = torch.add(lg_node_feat, m_feat)

        
        if len(self.block_2_s) > 0:
            for i in range(len(self.block_2_s)):
                layer = self.block_2_s[i]
                if i == 0:
                    l_feat, g_edge_feat = layer(g, g_node_feat, m_feat)
                if i == 1:
                    l_feat, g_edge_feat = layer(g, l_feat, g_edge_feat)
            l_feat = torch.add(g_node_feat, l_feat)

        l_feat = nn.functional.leaky_relu(torch.add(g_node_feat, l_feat))

        if len(self.block_3_s) > 0:
            for layer in self.block_3_s:
                m_feat = layer(g, lg, g_node_feat, m_feat, lg_edge_feat)
            m_feat = torch.add(lg_node_feat, m_feat)
        m_feat = nn.functional.leaky_relu(torch.add(lg_node_feat, m_feat))
        
        return l_feat, m_feat, fg_node_feat
        '''
    
        '''
        ###并行
        if len(self.block_0_s) > 0:
            for layer in self.block_0_s:
                x_feat = layer(lg, fg, lg_node_feat,fg_node_feat,fg_edge_feat)
            x_feat = torch.add(fg_node_feat, x_feat)


        # from H information to L information
        if len(self.block_1_s) > 0:
            for layer in self.block_1_s:
                m_feat = layer(g, lg, g_node_feat,lg_node_feat, fg_node_feat)
            m_feat = torch.add(lg_node_feat, m_feat)

        
        if len(self.block_2_s) > 0:
            for i in range(len(self.block_2_s)):
                layer = self.block_2_s[i]
                if i == 0:
                    l_feat, g_edge_feat = layer(g, g_node_feat, lg_node_feat)
                if i == 1:
                    l_feat, g_edge_feat = layer(g, l_feat, g_edge_feat)
            l_feat = torch.add(g_node_feat, l_feat)


        l_feat = nn.functional.leaky_relu(torch.add(g_node_feat, l_feat))

        if len(self.block_3_s) > 0:
            for layer in self.block_3_s:
                m_feat = layer(g, lg, g_node_feat, m_feat, lg_edge_feat)
            m_feat = torch.add(lg_node_feat, m_feat)
        m_feat = nn.functional.leaky_relu(torch.add(lg_node_feat, m_feat))
        
        if len(self.block_4_s) > 0:
            for layer in self.block_4_s:
                x_feat = layer(lg, fg, lg_node_feat, x_feat, fg_edge_feat)

        x_feat = nn.functional.leaky_relu(torch.add(fg_node_feat, x_feat))
        return l_feat, m_feat, x_feat
        '''
        
        '''
        # from H information to L information
        if len(self.block_0_s) > 0:
            for layer in self.block_0_s:
                x_feat = layer(lg, fg, lg_node_feat,fg_node_feat,fg_edge_feat)
            x_feat = torch.add(fg_node_feat, x_feat)

        if len(self.block_1_s) > 0:
            for layer in self.block_1_s:
                m_feat = layer(g, lg, g_node_feat,lg_node_feat, x_feat)
            m_feat = torch.add(lg_node_feat, m_feat)

        
        if len(self.block_2_s) > 0:
            for i in range(len(self.block_2_s)):
                layer = self.block_2_s[i]
                if i == 0:
                    l_feat, g_edge_feat = layer(g, g_node_feat, m_feat)
                if i == 1:
                    l_feat, g_edge_feat = layer(g, l_feat, g_edge_feat)
            l_feat = torch.add(g_node_feat, l_feat)

        l_feat = nn.functional.leaky_relu(torch.add(g_node_feat, l_feat))

        return l_feat, m_feat, x_feat
        '''

        
        '''
        # from L information to H information
        if len(self.block_2_s) > 0:
            for i in range(len(self.block_2_s)):
                layer = self.block_2_s[i]
                if i == 0:
                    l_feat, g_edge_feat = layer(g, g_node_feat, lg_node_feat)
                if i == 1:
                    l_feat, g_edge_feat = layer(g, l_feat, g_edge_feat)
            l_feat = torch.add(g_node_feat, l_feat)

        l_feat = nn.functional.leaky_relu(torch.add(g_node_feat, l_feat))

        if len(self.block_3_s) > 0:
            for layer in self.block_3_s:
                m_feat = layer(g, lg, l_feat, lg_node_feat, lg_edge_feat)
            m_feat = torch.add(lg_node_feat, m_feat)
        m_feat = nn.functional.leaky_relu(torch.add(lg_node_feat, m_feat))
        
        if len(self.block_4_s) > 0:
            for layer in self.block_4_s:
                x_feat = layer(lg, fg, m_feat, fg_node_feat, fg_edge_feat)
            #x_feat = torch.add(fg_node_feat, x_feat)

        x_feat = nn.functional.leaky_relu(torch.add(fg_node_feat, x_feat))

        return l_feat, m_feat, x_feat
        '''
        




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

        #self.lh_dir = HL_LH_block_2(in_feats, hidden_size, out_feats, encode_dim)

        self.block_layers.append(self.hl_dir)
        #self.block_layers.append(self.lh_dir)

        


    def forward(self, g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat,tras_med):

        # from H information to L information

        if tras_med == 'hl':
            for i in range(len(self.block_layers)):
                layer = self.block_layers[i]
                if i == 0:
                    g_node_feat,lg_node_feat,fg_node_feat = layer(g, lg, fg, device, g_node_feat,g_edge_feat,lg_node_feat,lg_edge_feat,fg_node_feat,fg_edge_feat)

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

        return g_node_feat,lg_node_feat,fg_node_feat
    
    
class SSP(nn.Module):
    def forward(self, x):
        return torch.log(0.5 * torch.exp(x) + 0.5)



class PCNN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, out_dim, tras_med, num_blcok,num_heads,num_layers,activation=swish,dropout = 0.1):
        super(PCNN, self).__init__()
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
        for i in range(self.num_blcok):
            self.blockconv.append(HL_LH_block(in_feats=self.in_feats, hidden_size=self.hidden_size, out_feats=self.hidden_size, encode_dim=self.encode_dim,num_heads=self.num_heads, num_layers=self.num_layers))
    
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.celu = nn.CELU()
        self.plus = nn.Softplus()

        self.norm = nn.BatchNorm1d(self.hidden_size*2)
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()


        layers4 = [
            nn.Linear(self.encode_dim[0]+self.encode_dim[1]+self.encode_dim[2], self.out_feats),
            self.leaky_relu,
            #nn.Dropout(dropout),
            nn.Linear(self.out_feats, self.out_dim),
            nn.Sigmoid()
        ]
        self.Readout = nn.Sequential(*layers4)

        
  

    def forward(self, g_feats, g_graph, lg_graph, fg_graph, g_graph_node_feat,g_graph_edge_feat, lg_graph_node_feat, lg_graph_edge_feat,fg_graph_node_feat,fg_graph_edge_feat, device, resent, pooling):
        
        batch_outputs = []
        for i in range(self.num_blcok):
            bolck = self.blockconv[i]
            out_1,out_2,out_3 = bolck(g_graph,lg_graph,fg_graph, device, g_graph_node_feat,g_graph_edge_feat, lg_graph_node_feat, lg_graph_edge_feat, fg_graph_node_feat,fg_graph_edge_feat,self.tras_med)

        if resent:

            out_1 = out_1 + g_graph_node_feat
            out_2 = out_2 + lg_graph_node_feat
            out_3 = out_3 + fg_graph_node_feat
            


        if pooling == 'avg':
            y1 = self.avgpool(g_graph, out_1)
            y2 = self.avgpool(lg_graph, out_2)
            y3 = self.avgpool(fg_graph, out_3)
            
        elif pooling == 'max':
            y1 = self.maxpool(g_graph, out_1)
            y2 = self.maxpool(lg_graph, out_2)
            y3 = self.maxpool(fg_graph, out_3)
        
        elif pooling == 'sum':
            y1 = self.sumpool(g_graph, out_1)
            y2 = self.sumpool(lg_graph, out_2)
            y3 = self.sumpool(fg_graph, out_3)


        else:
            print('No pooling found!!!!')

        
     
        
        y = torch.cat((y1,y2,y3), dim=1)
        batch_out = y.squeeze() 
        out = self.Readout(batch_out)
        
        

        return out
    
    def get_grad_norm_weights(self) -> nn.Module:
        return self.parameters()
