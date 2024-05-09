import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv,GCNConv
from torch_geometric.nn import global_max_pool as gmp
import pandas as pd
import numpy as np

# GCN  model
class GCNNet(torch.nn.Module):
    def __init__(self, num_features_xd=62, n_output=1, output_dim=64, dropout=0.2, num_net=5, file=None):
        super(GCNNet, self).__init__()
             
        self.num_net = num_net
        self.drug1_gcn1 = nn.ModuleList([GCNConv(num_features_xd, num_features_xd) for i in range(num_net)])
        self.drug1_gcn2 = nn.ModuleList([GCNConv(num_features_xd, num_features_xd) for i in range(num_net)])
        self.drug1_gcn3 = nn.ModuleList([GCNConv(num_features_xd, num_features_xd) for i in range(num_net)])

        self.drug1_fc_g1 = nn.ModuleList([nn.Linear(num_features_xd, output_dim) for i in range(num_net)]) 
        

        # combined layers
        self.fc1 = nn.Linear(output_dim, 64)
        self.out = nn.Linear(64, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(self, data_data, device):
        x11 = torch.Tensor().to(device)
        for i, module in enumerate(zip(self.drug1_gcn1,self.drug1_gcn2,self.drug1_gcn3,self.drug1_fc_g1)):
        
#        for i, module in enumerate(zip(self.drug1_gcn1,self.drug1_fc_g1)):
            data11 = data_data[i].to(device)
            
            x1, edge_index1, batch1 = data11.x, data11.edge_index, data11.batch

            drug1_gcn1 = module[0]
            drug1_gcn2 = module[1]
            drug1_gcn3 = module[2]
            drug1_fc_g1 = module[3]
            
            x1 = drug1_gcn1(x1, edge_index1)
            x1 = self.relu(x1)
            x1 = drug1_gcn2(x1, edge_index1)
            x1 = self.relu(x1)
#            x1 = drug1_gcn3(x1, edge_index1)
#            x1 = self.relu(x1)
    
            batch1 = batch1.type(torch.LongTensor)
            x1 = gmp(x1.to(batch1.device), batch1)         # global max pooling
            x1 = x1.to(drug1_fc_g1.weight.device)
            x1 = drug1_fc_g1(x1)
            x1 = self.relu(x1)
            x1 = self.dropout(x1)
                
            if i == 0:
                x11 = x1 * 1/self.num_net
            else:
                x11 = x11 + x1 * 1/self.num_net

        x11 = F.normalize(x11, 2, 1)

        xc = self.fc1(x11)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        out = self.out(xc)
        out = self.sigmoid(out)
        return out

