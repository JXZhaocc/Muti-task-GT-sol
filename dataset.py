import torch
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os, sys ,inspect
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.transforms import BaseTransform
import pyximport
add_path=os.path.realpath(__file__)
pyximport.install(setup_args={'include_dirs': np.get_include()})
from src import algos

from data_util import *
from torch_geometric.data import InMemoryDataset
import os.path as osp


def fn(data_list):
    max_num_nodes = max([data.sph.shape[0] for data in data_list])
    for data in data_list:
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        data.sph = torch.nn.functional.pad(data.sph, (0, pad_size, 0, pad_size), value=510)
    batched_data = Batch.from_data_list(data_list)
    return batched_data

class NewDataset_w_sph(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.dataset = original_dataset
        self.sph = []
        self.get_sph_all()
        assert len(self.sph)== len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        data['sph'] = self.sph[index]
        return data

    def __len__(self):
        return len(self.dataset)
    def get_sph_all(self):
        self.sph = []
        file = osp.join('/'.join(self.dataset.processed_paths[0].split('/')[:-1]), 'sph.pkl')
        if not os.path.exists(file):
            print('pre-process sph start!')
            progress_bar = tqdm(desc='pre-processing Data', total=len(self.dataset), ncols=40)
            for i in range(len(self.dataset)):
                self.process(i)
                progress_bar.update(1)
            progress_bar.close()
            pickle.dump(self.sph, open(file, 'wb'))
            print('pre-process sph done!')
        else:
            self.sph = pickle.load(open(file, 'rb'))
            print('load sph done!')

    def process(self, index):
        data = self.dataset[index]
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        self.sph.append(sp)
        """
        self.sph.append(adj_pinv(data, topk=10)) # ectd
        mat = cosine_similarity(data.x, data.x)
        mat = 1 / mat
        self.sph.append(mat) # cosine similarity
        """

class custom_InMemoryDataset(InMemoryDataset):
    def __init__(self, df=None, df_file=None, smiles='smiles',properties= ['A','B'] ,root='temp_datasets', smiles2graph=smiles2graph,
                 transform=None, pre_transform=None, force_reload=True):
        self.df_file = df_file
        self.df = df
        if self.df_file: 
            self.df = pd.read_csv(self.df_file)
            df.reset_index(drop=True, inplace=True)
        assert isinstance(self.df,pd.core.frame.DataFrame), 'provide pandas dataframe'
        assert smiles in self.df.columns , f'{smiles} is not in the columns of {self.df}'
        self.smiles= smiles
        self.properties=properties
        for pro in self.properties:
            assert pro in self.df.columns , f'{pro} is not in the columns of df'
    
        #self.root = root
        self.root= osp.join(root)
        self.smiles2graph = smiles2graph
        super().__init__(root=self.root,  transform = transform, pre_transform=pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return df_file

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def process(self):
        smiles_list = self.df[self.smiles]

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            try:            
                data = Data()
                smiles = smiles_list[i]
                graph = self.smiles2graph(smiles)
                assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert (len(graph['node_feat']) == graph['num_nodes'])
    
                data.__num_nodes__ = int(graph['num_nodes'])
                data.edge_index = torch.from_numpy(graph['edge_index']).to(
                    torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                    torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                #data.y = torch.Tensor([y])
                #y = data_df.iloc[i][properties]
                for pro in self.properties: data[pro]=self.df.iloc[i][pro]
                ########
                """
                N = data.x.shape[0]
                adj = torch.zeros([N, N])
                adj[data.edge_index[0, :], data.edge_index[1, :]] = True
                data.sph, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
                """
                ############
                data=self.pre_transform(data)
                data_list.append(data)
            except:
                print('escape the following smiles which can not be convert into a graph', smiles)

        #if self.pre_transform is not None:
        #    print('running  pre_transformation .. ')
        #    data_list = [self.pre_transform(data) for data in data_list]
            
        print('Saving...')
        #data, slices = self.collate(data_list)
        #torch.save((data, slices), self.processed_paths[0])
        self.save(data_list, self.processed_paths[0])

class infer_Dataset(torch.utils.data.Dataset):
    def __init__(self, smiles, smiles2graph,pre_transform=None ):
        super().__init__()
        self.smiles = smiles
        self.smiles2graph=smiles2graph
        self.pre_transform=pre_transform
        
    def __getitem__(self, index):
        s=self.smiles[index]
        data = Data()
        graph = self.smiles2graph(s)
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
        if self.pre_transform: data=self.pre_transform(data)
        return data

    def __len__(self):
        return len(self.smiles)

def prepare_batch_w_sph(data_list):
    max_num_nodes = max([data.sph.shape[0] for data in data_list])
    for data in data_list:
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        data.sph = torch.nn.functional.pad(data.sph, (0, pad_size, 0, pad_size), value=510)
    batched_data = Batch.from_data_list(data_list)
    return batched_data
