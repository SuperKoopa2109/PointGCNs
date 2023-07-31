# IMPORTS 


# import torch
# import torch_geometric
from torch import nn
from torch_geometric import nn as gnn
import torch_geometric.transforms as T
from torch.nn import functional as F

from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader

import numpy as np

from tqdm import tqdm


# MODELS

class generic_class():
    def __init__(self, something):
        self.sth = something

    def read(self):
        return self.sth

class generic_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, class_num=6, norm='None'):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num

        if norm == 'Batch':
            self.node_embeder = nn.Sequential(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm2d(hidden_dim, eps=1e-05, momentum=0.1), #, affine=True, track_running_stats=True, device=None, dtype=None)
                    nn.ReLU(inplace = True), # leaky ReLU instead?? 
                    nn.Linear(hidden_dim, embed_dim),
                    nn.BatchNorm2d(embed_dim, eps=1e-05, momentum=0.1), #, affine=True, track_running_stats=True, device=None, dtype=None)
                    nn.ReLU(inplace = True),
                    nn.Linear(embed_dim, class_num)
                ]
            )

        elif norm == 'Instance':
            self.node_embeder = nn.Sequential(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.InstanceNorm2d(hidden_dim, eps=1e-05, momentum=0.1), #, affine=False, track_running_stats=False, device=None, dtype=None)
                    nn.ReLU(inplace = True), # leaky ReLU instead?? 
                    nn.Linear(hidden_dim, embed_dim),
                    nn.InstanceNorm2d(embed_dim, eps=1e-05, momentum=0.1), #, affine=False, track_running_stats=False, device=None, dtype=None)
                    nn.ReLU(inplace = True),
                    nn.Linear(embed_dim, class_num)
                ]
            )

        else:
            self.node_embeder = nn.Sequential(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(inplace = True), # leaky ReLU instead?? 
                    nn.Linear(hidden_dim, embed_dim),
                    nn.ReLU(inplace = True),
                    nn.Linear(embed_dim, class_num)
                ]
            )
    
    def forward(self, data):
        
        return self.node_embeder(data['pos'])


class SAGE_model(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, no_of_layers=4, class_num=4, norm='None'):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        
# TODO: *** ACTIVATION FUNCTIONS
# TODO: for first activation function maybe only leaky ReLU makes sense
# TODO: Hyperparameters for non-linear activation function -> possibly for final sigmoid or tanh
# Tune on validation data

# TODO: Hyper Param: Type of Conv Layer -> GAT / SAGE / GIN

# TODO: Hyper Param: No of Conv Layer -> Repeating

# TODO: add stochastic regularization -> Dropout?? Batch/Graph-Normalization?? 

# TODO: Decide which target the hyper param optimization is supposed to optimise

# TODO: Evaluation step: maybe use better metrics than accuracy
        

        #Alternatively do this differently.. Apply layer for layer? 

        modules = []

        # embed data
        modules.append((gnn.SAGEConv(input_dim, hidden_dim), 'x, edge_index -> x'))
        modules.append(nn.ReLU(inplace=True))
        #modules.extend(self.get_hidden_layer(input_dim=input_dim, hidden_dim=hidden_dim, norm=norm))

        for layer_idx in range(1, no_of_layers):
            
            input_dim_layer = layer_idx * hidden_dim
            layer_dim = (layer_idx + 1) * hidden_dim

            modules.extend(self.get_hidden_layer(input_dim=input_dim, hidden_dim=layer_dim, norm=norm))

        modules.append(nn.Linear(input_dim, class_num))#(no_of_layers * hidden_dim, class_num))
        modules.append(nn.Sigmoid())

        self.node_embedder = gnn.Sequential(
                'x, edge_index',
                modules
                )

            # modules.append((gnn.SAGEConv(input_dim, hidden_dim), 'x, edge_index -> x'))

            # if norm == 'Batch':
            #     modules.append(gnn.BatchNorm(in_channels=hidden_dim, eps=1e-05, momentum=0.1))
            # elif norm == 'Instance':    
            #     modules.append(gnn.InstanceNorm(in_channels=hidden_dim, eps=1e-05, momentum=0.1))
            
            # modules.append(nn.ReLU(inplace=True))

            # gnn.Sequential('x, edge_index',
            #                modules)


        # if norm == 'Batch':
        #     self.node_embedder = gnn.Sequential(
        #         'x, edge_index',
        #         [
        #             (gnn.SAGEConv(input_dim, hidden_dim), 'x, edge_index -> x'),
        #             gnn.BatchNorm(in_channels=hidden_dim, eps=1e-05, momentum=0.1), #, affine: bool = True, track_running_stats: bool = True, allow_single_element: bool = False)),
        #             #nn.BatchNorm2d(hidden_dim, eps=1e-05, momentum=0.1), #, affine=True, track_running_stats=True, device=None, dtype=None)
        #             nn.ReLU(inplace = True), # leaky ReLU instead?? 
        #             (gnn.SAGEConv(hidden_dim, embed_dim), 'x, edge_index -> x'),
        #             gnn.BatchNorm(in_channels=embed_dim, eps=1e-05, momentum=0.1), #, affine: bool = True, track_running_stats: bool = True, allow_single_element: bool = False)),
        #             #nn.BatchNorm2d(embed_dim, eps=1e-05, momentum=0.1), #, affine=True, track_running_stats=True, device=None, dtype=None)
        #             nn.ReLU(inplace = True),
        #             nn.Linear(embed_dim, class_num)
        #             #nn.ReLU(inplace = True)
        #         ]
        #     )

        # elif norm == 'Instance':
        #     self.node_embedder = gnn.Sequential(
        #         'x, edge_index',
        #         [
        #             (gnn.SAGEConv(input_dim, hidden_dim), 'x, edge_index -> x'),
        #             gnn.InstanceNorm(in_channels=hidden_dim, eps=1e-05, momentum=0.1), #, affine: bool = False, track_running_stats: bool = False)
        #             #nn.InstanceNorm2d(hidden_dim, eps=1e-05, momentum=0.1), #, affine=False, track_running_stats=False, device=None, dtype=None)
        #             nn.ReLU(inplace = True), # leaky ReLU instead?? 
        #             (gnn.SAGEConv(hidden_dim, embed_dim), 'x, edge_index -> x'),
        #             gnn.InstanceNorm(in_channels=embed_dim, eps=1e-05, momentum=0.1), #, affine: bool = False, track_running_stats: bool = False)
        #             #nn.InstanceNorm2d(hidden_dim, eps=1e-05, momentum=0.1), #, affine=False, track_running_stats=False, device=None, dtype=None)
        #             nn.ReLU(inplace = True),
        #             nn.Linear(embed_dim, class_num)
        #             #nn.ReLU(inplace = True)
        #         ]
        #     )

        # else:
        #     self.node_embedder = gnn.Sequential(
        #         'x, edge_index',
        #         [
        #             (gnn.SAGEConv(input_dim, hidden_dim), 'x, edge_index -> x'),
        #             nn.ReLU(inplace = True), # leaky ReLU instead?? 
        #             (gnn.SAGEConv(hidden_dim, embed_dim), 'x, edge_index -> x'),
        #             nn.ReLU(inplace = True),
        #             nn.Linear(embed_dim, class_num)
        #             #nn.ReLU(inplace = True)
        #         ]
        #     )
        
        


        # *************************

        # self.aggr_1 = gnn.aggr.MeanAggregation()
        # self.mlp = nn.Sequential(
        #     nn.Linear(embed_dim, class_num),
        #     nn.ReLU(inplace=True),
        #     nn.Linear()
        # )
        
        
        
        
#         self.graph_embeder = gnn.Sequential(
#         'x, edge_index, batch',
#             [

#             ]
#         )

    def get_hidden_layer(self, input_dim=16, hidden_dim=64, norm='Batch'):
        modules = []

        modules.append((gnn.SAGEConv(input_dim, hidden_dim), 'x, edge_index -> x'))

        if norm == 'Batch':
            modules.append(gnn.BatchNorm(in_channels=hidden_dim, eps=1e-05, momentum=0.1))
        elif norm == 'Instance':    
            modules.append(gnn.InstanceNorm(in_channels=hidden_dim, eps=1e-05, momentum=0.1))

        modules.append(nn.ReLU(inplace=True))

        return modules
    # gnn.Sequential('x, edge_index',
    #                    modules)

    
    def forward(self, data):
        
#         x_embed = self.node_embeder(data['pos'], data['edge_index'])
        
        
        return self.node_embedder(data['x'], data['edge_index']) #self.mlp( self.aggr_1(x_embed, data['batch']) )