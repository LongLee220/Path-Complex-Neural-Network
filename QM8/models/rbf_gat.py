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
from models.activations import swish
from models.envelope import Envelope
from models.initializers import GlorotOrthogonal






device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")





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





class BesselBasisLayer(nn.Module):
    def __init__(self,
                 num_radial,
                 cutoff,
                 envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()
        
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.frequencies = nn.Parameter(torch.Tensor(num_radial))
        self.reset_params()

    def reset_params(self):
        with torch.no_grad():
            torch.arange(1, self.frequencies.numel() + 1, out=self.frequencies).mul_(np.pi)
        self.frequencies.requires_grad_()

    def forward(self, g):
        d_scaled = g.edata['feat'] / self.cutoff
        # Necessary for proper broadcasting behaviour
        d_scaled = torch.unsqueeze(d_scaled, -1)
        d_cutoff = self.envelope(d_scaled)
        g.edata['rbf'] = d_cutoff * torch.sin(self.frequencies * d_scaled)
        return g





class SphericalBasisLayer(nn.Module):
    def __init__(self,
                 num_spherical,
                 num_radial,
                 cutoff,
                 envelope_exponent=5):
        super(SphericalBasisLayer, self).__init__()

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        # retrieve formulas
        self.bessel_formulas = bessel_basis(num_spherical, num_radial)  # x, [num_spherical, num_radial] sympy functions
        self.sph_harm_formulas = real_sph_harm(num_spherical)  # theta, [num_spherical, ] sympy functions
        self.sph_funcs = []
        self.bessel_funcs = []

        # convert to torch functions
        x = sym.symbols('x')
        theta = sym.symbols('theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                first_sph = sym.lambdify([theta], self.sph_harm_formulas[i][0], modules)(0)
                self.sph_funcs.append(lambda tensor: torch.zeros_like(tensor) + first_sph)
            else:
                self.sph_funcs.append(sym.lambdify([theta], self.sph_harm_formulas[i][0], modules))
            for j in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], self.bessel_formulas[i][j], modules))

    def get_bessel_funcs(self):
        return self.bessel_funcs

    def get_sph_funcs(self):
        return self.sph_funcs
    



class EmbeddingBlock_LG(nn.Module):
    def __init__(self,
                 emb_size,
                 num_radial,
                 bessel_funcs,
                 cutoff,
                 envelope_exponent,
                 num_atom_types=95,
                 activation=None):
        super(EmbeddingBlock_LG, self).__init__()

        self.bessel_funcs = bessel_funcs
        self.cutoff = cutoff
        self.activation = activation
        self.envelope = Envelope(envelope_exponent)
        self.embedding = nn.Embedding(num_atom_types, emb_size)
        self.dense_rbf = nn.Linear(num_radial, emb_size)
        self.dense = nn.Linear(emb_size * 3, emb_size)
        self.reset_params()
    
    def reset_params(self):
        nn.init.uniform_(self.embedding.weight, a=-np.sqrt(3), b=np.sqrt(3))
        GlorotOrthogonal(self.dense_rbf.weight)
        GlorotOrthogonal(self.dense.weight)

    def edge_init(self, edges):
        """ msg emb init """
        # m init
        rbf = self.dense_rbf(edges.data['rbf'])
        if self.activation is not None:
            rbf = self.activation(rbf)

        m = torch.cat([edges.src['feat'], edges.dst['feat'], rbf], dim=-1)
        m = self.dense(m)
        if self.activation is not None:
            m = self.activation(m)
        
        # rbf_env init
        d_scaled = edges.data['feat'] / self.cutoff
        rbf_env = [f(d_scaled) for f in self.bessel_funcs]
        rbf_env = torch.stack(rbf_env, dim=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf_env

        return {'m': m, 'rbf_env': rbf_env}

    def forward(self, g):
        g.ndata['feat'] = self.embedding(g.ndata['Z'])
        g.apply_edges(self.edge_init)
        return g
    




class EmbeddingBlock_GG(nn.Module):
    def __init__(self,
                 emb_size,
                 num_radial,
                 bessel_funcs,
                 cutoff,
                 envelope_exponent,
                 num_atom_types=95,
                 activation=None):
        super(EmbeddingBlock_GG, self).__init__()

        self.bessel_funcs = bessel_funcs
        self.cutoff = cutoff
        self.activation = activation
        self.envelope = Envelope(envelope_exponent)
        self.embedding = nn.Embedding(num_atom_types, emb_size)
        self.dense_rbf = nn.Linear(num_radial, emb_size)
        self.dense = nn.Linear(emb_size * 3, emb_size)
        self.reset_params()
    
    def reset_params(self):
        nn.init.uniform_(self.embedding.weight, a=-np.sqrt(3), b=np.sqrt(3))
        GlorotOrthogonal(self.dense_rbf.weight)
        GlorotOrthogonal(self.dense.weight)

    def edge_init(self, edges):
        """ msg emb init """
        # m init
        rbf = self.dense_rbf(edges.data['rbf'])
        if self.activation is not None:
            rbf = self.activation(rbf)

        m = torch.cat([edges.src['feat'], edges.dst['feat'], rbf], dim=-1)
        m = self.dense(m)
        if self.activation is not None:
            m = self.activation(m)
        
        # rbf_env init
        d_scaled = edges.data['feat'] / self.cutoff
        rbf_env = [f(d_scaled) for f in self.bessel_funcs]
        rbf_env = torch.stack(rbf_env, dim=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf_env

        return {'m': m, 'rbf_env': rbf_env}

    def forward(self, g):
        g.ndata['feat'] = self.embedding(g.ndata['Z'])
        g.apply_edges(self.edge_init)
        return g
    


    




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
    
    



class RBF_GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, encode_dim, out_dim,dropout=0.2, num_blcok=1,activation=swish):
        super(RBF_GAT, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.out_feats = out_feats
        self.num_blcok = num_blcok
        self.out_dim = out_dim
        self.activation = activation
        self.atom_feature_size = encode_dim[0]
        self.bond_feature_size = encode_dim[1]
        self.blockconv = nn.ModuleList()

        
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

        
    def forward(self, lg, gg, device, resent, pooling):
        
        
        batch_outputs = []
        
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
    
