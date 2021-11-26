import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.utils.data 
from torch_geometric.data import DataLoader
from torch_geometric.data import Data

from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from sklearn.metrics import f1_score, accuracy_score


class GDataset(Dataset):
    def __init__(self, nodes, edges, y):
        super(GDataset, self).__init__()
        
        self.nodes = nodes
        self.edges = edges
        self.y = y
        
    def __getitem__(self, idx):
        edge_index = torch.tensor(self.edges[idx].T, dtype=torch.long)
        x = torch.tensor(self.nodes[idx], dtype=torch.float)
        y = torch.tensor(self.y[idx], dtype=torch.float)
        return Data(x=x,edge_index=edge_index,y=y)
    
    def __len__(self):
        return self.y.shape[0]
    
    def collate_fn(self,batch):
        pass
    
    
class GAT(torch.nn.Module):
    def __init__(self, in_embd, layer_embd, out_embd, heads, dropout):
        super(GAT, self).__init__()
        
        self.GATConv1 = GATConv(in_embd, layer_embd, heads, concat=False, dropout = dropout)
        self.GATConv2 = GATConv(layer_embd, out_embd, heads, concat=False, dropout = dropout)
        
        self.pool = GlobalAttention(gate_nn=nn.Sequential( \
                nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)))
        
        self.graph_linear = nn.Linear(out_embd, 1)
        
    def forward(self, x, edge_index, batch):
      out = self.GATConv1(x, edge_index)
      out = F.sigmoid(out)
      out = self.GATConv2(out, edge_index)
      out = F.sigmoid(out)
      out = self.pool(out, batch)
      out = self.graph_linear(out)
      out = F.sigmoid(out)
      return out

def train(EPOCH, split, batch_size):
    nodes = np.load('nodes.npy', allow_pickle=True)
    edges = np.load('edges.npy', allow_pickle=True)
    y = np.load('y.npy', allow_pickle=True)
    
    train_num = int(y.shape[0]*split)
    test_num = y.shape[0] - train_num
    dataset = GDataset(nodes, edges, y)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_num, shuffle=False)

    model = GAT(in_embd=6, layer_embd=64, out_embd=256, heads=4, dropout=0.0)

    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

    criterion = torch.nn.BCELoss()

    for epoch in range(EPOCH):
        print('epoch {}:'.format(epoch))
        preds = []
        labels = []
        for data in train_loader:
            pred = model(data.x, data.edge_index, data.batch)
            loss = criterion(pred.squeeze(), data.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            preds.append(((pred.detach().numpy())>0.5)*1)
            labels.append(data.y.detach().numpy())
            
        preds  = np.concatenate(preds)
        preds = np.array(preds)
        labels  = np.concatenate(labels)
        labels = np.array(labels)
            
        print('\ttrain acc = {}'.format(round(accuracy_score(labels, preds)*100, 2)))
        
        for data in test_loader:
            preds = (model(data.x, data.edge_index, data.batch).detach().numpy()>0.5)*1
            labels = data.y.detach().numpy()
            
        print('\ttest acc = {}'.format(round(accuracy_score(labels, preds)*100, 2)))

if __name__ == '__main__':
    train(EPOCH=10, split=0.7, batch_size=32)
