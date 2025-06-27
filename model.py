import torch
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential, Parameter, LeakyReLU, BatchNorm1d
import numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import math, copy ,os ,inspect
import torch_geometric.transforms as T
from torch_geometric.nn import global_add_pool, GINConv, GINEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch, InMemoryDataset
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos
from grad_conv import GPSConv
from utils import process_hop
from data_util import get_atom_feature_dims, get_bond_feature_dims 
from dataset import infer_Dataset, prepare_batch_w_sph
from torch.utils.data import  DataLoader, Dataset
from torch import optim
from torch_geometric.utils import to_dense_batch
from grad_conv import MultiheadAttention

import copy

from data_util import smiles2graph



class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, optional_full_atom_features_dims=None):
        super(AtomEncoder, self).__init__()


        self.atom_embedding_list = torch.nn.ModuleList()

        if optional_full_atom_features_dims is not None:
            full_atom_feature_dims = optional_full_atom_features_dims
        else:
            full_atom_feature_dims = get_atom_feature_dims()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding

class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        full_bond_feature_dims = get_bond_feature_dims()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   


def process_hop(sph, gamma, hop, slope=0.1):
    leakyReLU = LeakyReLU(negative_slope=slope)
    sph = sph.unsqueeze(1)
    sph = sph - hop
    sph = leakyReLU(sph)
    sp = torch.pow(gamma, sph)
    return sp
 
'''def Smiles2GraphBatch(smiles, smiles2graph ):
    data_list = []
    for s in smiles:    
        data = Data()
        graph = smiles2graph(s)
        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert (len(graph['node_feat']) == graph['num_nodes'])
        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        data['sph']=sp
        data_list.append(data)
    max_num_nodes = max([data.sph.shape[0] for data in data_list])
    for data in data_list:
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        data.sph = torch.nn.functional.pad(data.sph, (0, pad_size, 0, pad_size), value=510)
    batched_data = Batch.from_data_list(data_list)
    return batched_data   '''

class prediction_layer(torch.nn.Module):
    def __init__(self, d_model,
                 activation=None,leaky_relu_slope=0.01, dropout=0.1,out_dim=None ):
        super(prediction_layer, self).__init__()
        if not out_dim:
            out_dim= d_model
        self.linear=torch.nn.Linear(d_model, out_dim)
        self.activation= activation
        if not self.activation:
            self.activation=torch.nn.LeakyReLU(leaky_relu_slope)
        self.LayerNorm = torch.nn.LayerNorm(out_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self, hidden):
        out = self.activation(self.linear(hidden))
        return self.dropout(self.LayerNorm(out))

class regression_head(torch.nn.Module):
    def __init__(self, task_name,hidden_size=250,pred_hidden_dropout=0.1,
     n_layers=1, n_output=1 ,out_dim=None ):
        super(regression_head, self).__init__()
        if not out_dim:
            out_dim= hidden_size
        self.task_name=task_name
        self.n_output=n_output
        if n_layers > 1:
            layer=prediction_layer(d_model=hidden_size, dropout=pred_hidden_dropout)
            layers=[copy.deepcopy(layer) for i in range(n_layers-1)]
        else:
            layers=[]
        las_layer=prediction_layer(d_model=hidden_size, dropout=pred_hidden_dropout, out_dim=out_dim)
        layers= layers+[las_layer]
        self.layers= torch.nn.ModuleList(layers)
        self.activation = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(pred_hidden_dropout)
        self.linear=torch.nn.Linear(out_dim, n_output)
        self.uncertainty  = torch.nn.Parameter(torch.zeros(1))
    def forward(self, hidden):
        for  layer in self.layers:
            hidden= layer(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self.linear(hidden)
    def loss_fn(self, y_pred, y_true):
        loss_fuc=torch.nn.MSELoss(reduction='none')
        y_pred=y_pred.contiguous().view(-1)#, self.n_output)
        y_true = y_true.contiguous().view(-1).type(y_pred.dtype)
        # Missing data are nan's
        mask = torch.isnan(y_true)
        #print(y_pred.size(),y_true.size(), mask.size())
        loss= loss_fuc(y_pred[~mask] ,y_true[~mask] )
        ## R2
        R2=1-(torch.sum((y_pred[~mask] - y_true[~mask]) ** 2) / torch.sum((y_true[~mask] - y_true[~mask].mean()) ** 2)).item()
        return loss ,R2

class Classification_head(torch.nn.Module):
    """
    N-class classification model
    """
    def __init__(self, task_name,n_layers=1,n_output=2,hidden_size=250,
                 pred_hidden_dropout=0.1,ignore_index=-100,labels_id2name=None, classes_weights=None) :
        super().__init__()
        self.task_name=task_name
        self.n_output=n_output
        self.ignore_index=ignore_index
        self.labels_id2name=labels_id2name
        if isinstance(classes_weights, (torch.Tensor)) :
            assert len(classes_weights) == n_output
            self.classes_weights=classes_weights.to(torch.float32)
        else:
            self.classes_weights=torch.ones(n_output)
        layer=prediction_layer(d_model=hidden_size, dropout=pred_hidden_dropout)
        self.layers= torch.nn.ModuleList([copy.deepcopy(layer) for i in range(n_layers)])
        self.linear=torch.nn.Linear(hidden_size, self.n_output)
        self.uncertainty  = torch.nn.Parameter(torch.zeros(1))
        #self.loss_fn =loss_fn
        #if not self.loss_fn:   self.loss_fn= nn.NLLLoss(ignore_index=-10000,reduction='none')
    def forward(self, hidden_states):
        hidden=hidden_states[0, :].unsqueeze(0)
        for  layer in self.layers:
            hidden= layer(hidden)
        out  = self.linear(hidden) # (T,B,n_class)
        return  F.log_softmax(out, dim=-1)
    def loss_fn(self, probs, tgt ):
        # probs (T,B,n_class)
        # tgt (T,B)
        #device= probs.device
        device=next(self.parameters()).device
        loss_fuc=torch.nn.NLLLoss(ignore_index=self.ignore_index,weight=self.classes_weights.to(device), reduction='none')
        probs_flatten=probs.view(-1, self.n_output)
        tgt_flatten = tgt.contiguous().view(-1)
        loss= loss_fuc(probs_flatten,tgt_flatten)#.to(device)
        ## acc
        mask=(tgt_flatten == self.ignore_index)
        predicted_labels = torch.argmax(probs_flatten, dim=-1)
        if mask.sum()>0:
            acc=100.0* (predicted_labels[~mask] == tgt_flatten[~mask]).sum().item()/len(tgt_flatten[~mask])
        else:
            acc=100.0* (predicted_labels == tgt_flatten).sum().item()/len(tgt_flatten)
        return loss ,acc

class Gradformer_encoder(torch.nn.Module):
    def __init__(self, channels=100,  node_dim=9, edge_dim=3, mpnn='GINE',
                 pe_norm=False ,pe_origin_dim=20,pe_dim=20 , gamma=0.5, slope=0.0,
                 pool='mean', node_method='ogb', edge_method='ogb', nhead=4,n_hop=5,num_layers=10, 
                 dropout=0.1, drop_prob=0.0, attn_dropout=0.1,  smiles2graph=smiles2graph, pre_transform=None, hops_list=None
                ):
        super().__init__()
        self.hidden_size=channels
        self.smiles2graph= copy.deepcopy(smiles2graph)
        self.pre_transform=pre_transform
        self.gamma = gamma
        self.slope = slope
        self.mpnn = mpnn
        self.pool = pool
        self.pe = pe_norm
        self.node_method = node_method
        self.edge_method = edge_method
        self.node_add = Linear(node_dim, channels)
        self.pe_add = Linear(pe_origin_dim, channels)
        self.pe_lin = Linear(pe_origin_dim, pe_dim)
        self.node_lin = Linear(node_dim, channels - pe_dim)
        self.no_pe = Linear(node_dim, channels)
        self.node_emb = Embedding(node_dim, channels - pe_dim)
        #self.atom_enc = AtomEncoder(args.channels - args.pe_dim - 1)
        self.atom_enc = AtomEncoder(channels - pe_dim )
        self.bond_enc = BondEncoder(channels)
        self.edge_emb = Embedding(edge_dim, channels)
        self.pe_norm = BatchNorm1d(pe_origin_dim)
        if hops_list:
            assert len(hops)== nheads
            self.hop = Parameter(torch.tensor(hops_list).reshape([nhead,1,1]))
        else:
            self.hop = Parameter(torch.full((nhead, 1, 1), float(n_hop)))
        self.convs = ModuleList()
        if mpnn == 'GIN':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(channels, channels),
                    ReLU(),
                    Linear(channels, channels),
                )
                conv = GPSConv(channels, GINConv(nn), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GINE':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(channels, channels),
                    ReLU(),
                    Linear(channels, channels),
                )
                conv = GPSConv(channels, GINEConv(nn), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GCN':
            Lap_pe = False#True if "hiv" in args.dataset else False
            for _ in range(num_layers):
                conv = GPSConv(channels, GatedGCNLayer(channels, channels, 0, True,
                                                            equivstable_pe=Lap_pe),
                               heads=nhead, dropout=dropout, attn_dropout=attn_dropout,
                               drop_prob=drop_prob)
                self.convs.append(conv)

    def forward(self, x, pe, edge_index, edge_attr, batch, sph):
        if self.pe:
            pe = self.pe_norm(pe)
        #print('x=',x.size(),'edge_index',edge_index.size(),'pe',pe.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        # process node
        if self.node_method == 'add':
            x = self.node_add(x) + self.pe_add(pe)
        if self.node_method == 'linear':
            x = torch.cat((self.node_lin(x), self.pe_lin(pe)), 1)
        if self.node_method == 'embedding':
            x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(pe)), 1)
        if self.node_method == 'ogb':
            x = torch.cat((self.atom_enc(x), self.pe_lin(pe)), 1)
        if self.node_method == 'no_pe':
            x = self.no_pe(x)
        #print('x=',x.size(),'edge_index',edge_index.size(),'pe',pe.size(),'edge_attr', edge_attr.size(),'sph',sph.size())

        # process edge
        if self.edge_method == 'ogb':
            edge_attr = self.bond_enc(edge_attr)
        if self.edge_method == 'embedding':
            edge_attr = self.edge_emb(edge_attr)

        # get the sph
        sph = process_hop(sph, self.gamma, self.hop, self.slope)
        #print('x=',x.size(),'edge_index',edge_index.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        # Conv & MPNN
        if self.mpnn == 'GIN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch)
        elif self.mpnn == 'GCN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=edge_attr, pe=pe)
        elif self.mpnn == 'GINE':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=edge_attr)
        #print("x before pooling=", x.size())
        # pooling
        if self.pool == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool == 'add':
            x = global_add_pool(x, batch)
        elif self.pool == 'max':
            x = global_max_pool(x, batch)
        return x

  
class Gradformer_DownstreamTasks(torch.nn.Module):
    def __init__(self, tasks, fine_tune_scale=None ,lr=0.0005 , encoder= 'Gradformer_pretrained'):
        super().__init__()
        
        if encoder== "new":
            self.encoder=  Gradformer_encoder(smiles2graph=smiles2graph, pre_transform= T.AddRandomWalkPE(walk_length=20, attr_name='pe') )
        else:
            raise NotImplementedError
        self.hidden_size=self.encoder.hidden_size #100
        
        if tasks:
            if type(tasks)==type({}):
                tasks=self.make_tasks(tasks)
        self.tasks   = torch.nn.ModuleList(tasks)
        self.n_tasks= 1+len(tasks)
        
        if fine_tune_scale:
            assert fine_tune_scale >0 and fine_tune_scale<=1 , "fine_tune_scale should be in [0-1]"
            print(f'the learning rate of the encoder is scaled by {fine_tune_scale}')
            parm_groups=[{ 'params': self.encoder.parameters(), 'lr': fine_tune_scale*lr }]
            parm_groups+=[{ 'params': self.tasks.parameters(), 'lr': lr}]
            self.optimizer = optim.Adam(parm_groups, lr=lr)
        else:
            print(f'freezing the parameters of the base model')
            for p in self.encoder.parameters(): p.requires_grad = False
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

    def forward(self, x, pe, edge_index, edge_attr, batch, sph):
        x= self.encoder( x, pe, edge_index, edge_attr, batch, sph)
        out={}
        #print("x before tasks=", x.size())
        if self.tasks:
            for task in self.tasks:
                out[task.task_name]=task(x)
        #print("out[task.task_name]", out[task.task_name].size())
        return out
    def make_tasks(self, tasks_dict):
        '''
        example:
        inputs: tasks_dict={
                    'chemaxon_LogD' :{'is_regression':1 , 'n_output':1,'n_layers': 1},
                    'pic50':{'is_regression':0 , 'n_output':1,'n_layers': 1}}
        output:
        tasks list
        '''
        hidden_size=self.hidden_size
        #if self.pool:        hidden_size=hidden_size
        tasks=[]
        for task_name in tasks_dict:
            try: pred_hidden_dropout=tasks_dict[task_name]['pred_hidden_dropout']
            except:pred_hidden_dropout=0.1
            try: out_dim=tasks_dict[task_name]['out_dim']
            except:out_dim=None
            if  tasks_dict[task_name]['is_regression']:
                model= regression_head(task_name=task_name,
                                        hidden_size=hidden_size,
                                      n_layers=tasks_dict[task_name]['n_layers'],
                                      n_output=tasks_dict[task_name]['n_output'],
                                      pred_hidden_dropout=pred_hidden_dropout, out_dim=out_dim)
                #tasks_dict[task_name]['loss_fn']= torch.nn.MSELoss(reduction='none')
            else:#Classification_head
                try:  ignore_index=tasks_dict[task_name]['mask']
                except:ignore_index= -100
                try: labels_id2name=tasks_dict[task_name]['labels_id2name']
                except: labels_id2name=None
                model= Classification_head(task_name=task_name,
                                        hidden_size=self.hidden_size,
                                      n_layers=tasks_dict[task_name]['n_layers'],
                                      n_output=tasks_dict[task_name]['n_output'],
                                      labels_id2name=labels_id2name,
                                      ignore_index=ignore_index,
                                      pred_hidden_dropout=pred_hidden_dropout)
            tasks.append(model)
        return tasks

    def truncate_tasks(self, tasks_to_keep):
            tasks =[]
            for task in self.tasks:
                if task.task_name in tasks_to_keep:
                    tasks.append(task)
            self.tasks=torch.nn.ModuleList(tasks)
   
    def predict(self, smiles,batch_size=50,prob=False,class_label_id=False, convert_log10=True):
        if isinstance(smiles, str): smiles=[smiles]
        if(len(smiles)>25):        
            return self.predict_largre_batch(smiles=smiles,batch_size=batch_size,
                                             prob=prob,class_label_id=class_label_id, convert_log10=convert_log10)
        else:        
            return self.predict_small_batch(smiles=smiles,batch_size=10,
                                             prob=prob,class_label_id=class_label_id, convert_log10=convert_log10)
        
    def predict_small_batch(self, smiles,batch_size=50,prob=False,class_label_id=False, convert_log10=True):
            device=next(self.parameters()).device
            outputs={}
            for task in self.tasks:
                if task.__class__.__name__ == 'regression_head':
                    outputs[f'predicted_{task.task_name}']=[]
                elif  task.__class__.__name__ == 'Classification_head':
                    if class_label_id: outputs[f'predicted_{task.task_name}']=[]
                    if prob: outputs[f'prob_{task.task_name}']=[]
                    outputs[f'predicted_class_{task.task_name}']=[]
            st=0
            end=0
            while end < (len(smiles)):
                end= min(st+batch_size, len(smiles))
                batch=self.Smiles2GraphBatch(smiles[st:end])
                st+=batch_size
                batch = batch.to( device)
                node_num = batch.sph.shape[-1]
                batch.sph = batch.sph.reshape(-1, node_num, node_num)
                if batch.edge_attr is None:
                    batch.edge_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
                self.eval()
                with torch.no_grad():
                    #out = self.model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, batch.sph)
                    #(self, x, pe, edge_index, edge_attr, batch, sph)
                    out= self.forward(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, batch.sph)
                for task in self.tasks:
                    if task.__class__.__name__ == 'regression_head':
                        if task.n_output ==1:
                            outputs[f'predicted_{task.task_name}'].append(out[task.task_name].view(-1).detach().cpu().numpy() )
                        else:
                            outputs[f'predicted_{task.task_name}'].append(out[task.task_name].view(-1, task.n_output).detach().cpu().numpy() )
                    elif task.__class__.__name__ == 'Classification_head':
                        probs_flatten=out[task.task_name].view(-1, task.n_output)
                        predicted_labels = torch.argmax(probs_flatten, dim=-1).view(-1).detach().cpu().numpy()
                        if class_label_id: outputs[f'predicted_{task.task_name}'].append(predicted_labels)
                        labels=  np.array([task.labels_id2name.get(i) for i in predicted_labels])
                        #outputs[f'predicted_{task.task_name}'].append(predicted_labels)
                        outputs[f'predicted_class_{task.task_name}'].append(labels)
                        if prob:
                            outputs[f'prob_{task.task_name}'].append(probs_flatten.detach().cpu().numpy())
            outputs = {key: np.concatenate(value, axis=0) for key, value in outputs.items()}
            if convert_log10:
                logk= [k for k in outputs.keys() if k.startswith('predicted_log10') ]
                if len(logk):
                    for p in logk:
                        tp='_'.join(p.split('_')[2:])
                        outputs[f'predicted_{tp}']=10**outputs[p]
                        outputs.pop(p)
            return outputs
    def Smiles2GraphBatch(self, smiles ):
        data_list = []
        for s in smiles:    
            data = Data()
            graph = self.encoder.smiles2graph(s)
            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])
            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            N = data.x.shape[0]
            adj = torch.zeros([N, N])
            adj[data.edge_index[0, :], data.edge_index[1, :]] = True
            sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
            data['sph']=sp
            if self.encoder.pre_transform: data=self.encoder.pre_transform(data)
            data_list.append(data)
        max_num_nodes = max([data.sph.shape[0] for data in data_list])
        for data in data_list:
            num_nodes = data.num_nodes
            pad_size = max_num_nodes - num_nodes
            data.sph = torch.nn.functional.pad(data.sph, (0, pad_size, 0, pad_size), value=510)
        batched_data = Batch.from_data_list(data_list)
        return batched_data
    def predict_largre_batch(self, smiles,batch_size=50,prob=False,class_label_id=False, convert_log10=True, num_workers=4):
            if isinstance(smiles, pd.core.series.Series ):
                smiles=smiles.tolist()
            device=next(self.parameters()).device
            outputs={}
            for task in self.tasks:
                if task.__class__.__name__ == 'regression_head':
                    outputs[f'predicted_{task.task_name}']=[]
                elif  task.__class__.__name__ == 'Classification_head':
                    if class_label_id: outputs[f'predicted_{task.task_name}']=[]
                    if prob: outputs[f'prob_{task.task_name}']=[]
                    outputs[f'predicted_class_{task.task_name}']=[]
            temp_dataset=infer_Dataset( smiles, smiles2graph=self.encoder.smiles2graph,pre_transform=self.encoder.pre_transform )
            loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, collate_fn=prepare_batch_w_sph, num_workers=num_workers)
            for batch in iter(loader):
                    batch = batch.to(device)
                    node_num = batch.sph.shape[-1]
                    batch.sph = batch.sph.reshape(-1, node_num, node_num)
                    if batch.edge_attr is None:
                        batch.edge_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
                    #out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
                    with torch.no_grad():
                        out= self.forward(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch, batch.sph)
                    for task in self.tasks:
                        if task.__class__.__name__ == 'regression_head':
                            if task.n_output ==1:
                                outputs[f'predicted_{task.task_name}'].append(out[task.task_name].view(-1).detach().cpu().numpy() )
                            else:
                                outputs[f'predicted_{task.task_name}'].append(out[task.task_name].view(-1, task.n_output).detach().cpu().numpy() )
                        elif task.__class__.__name__ == 'Classification_head':
                            probs_flatten=out[task.task_name].view(-1, task.n_output)
                            predicted_labels = torch.argmax(probs_flatten, dim=-1).view(-1).detach().cpu().numpy()
                            if class_label_id: outputs[f'predicted_{task.task_name}'].append(predicted_labels)
                            labels=  np.array([task.labels_id2name.get(i) for i in predicted_labels])
                            #outputs[f'predicted_{task.task_name}'].append(predicted_labels)
                            outputs[f'predicted_class_{task.task_name}'].append(labels)
                            if prob:
                                outputs[f'prob_{task.task_name}'].append(probs_flatten.detach().cpu().numpy())
            outputs = {key: np.concatenate(value, axis=0) for key, value in outputs.items()}
            if convert_log10:
                logk= [k for k in outputs.keys() if k.startswith('predicted_log10') ]
                if len(logk):
                    for p in logk:
                        tp='_'.join(p.split('_')[2:])
                        outputs[f'predicted_{tp}']=10**outputs[p]
                        outputs.pop(p)
            return outputs
