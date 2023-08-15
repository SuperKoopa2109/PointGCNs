# IMPORTS 


# import torch
# import torch_geometric
from torch import nn
from torch_geometric import nn as gnn
import torch_geometric.transforms as T
from torch.nn import functional as F
from torch_geometric.utils import dropout_node

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
    def __init__(self, 
                 input_dim, 
                 embed_dim, 
                 hidden_dim,
                 conv_type = 'SAGEConv', 
                 no_of_layers = 4, 
                 class_num = 4, 
                 drop_rate = 0.,
                 negative_slope = 0., 
                 norm = 'None'):
        """
            Model to predict node classification providing configuration of convolutional layer type
            
            input_dim: dimension of input [int]
            embed_dim: dimension of embedding [int]
            hidden_dim: dimension of first hidden layer -> scaled by amount of hidden layers for following hidden layers [int]
            conv_type: type of convolutional layer ['SAGEConv', 'GAT', 'GCN']
            no_of_layers: number of hidden layers (residual layers) [int]
            class_num: number of classes predicted for output linear layer [int]
            drop_rate: rate for dropout; set to 0. for not using dropout at all [float]
            negative_slope: rate for negative slope in leaky ReLU activation functions [float]
            norm: type of normalization used; set to 'None' or leave default for not using normalization at all ['None', 'Batch', 'Instance', 'Layer']
            
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.conv_type = conv_type
        self.no_of_layers = no_of_layers
        self.class_num = class_num
        self.drop_rate = drop_rate
        self.negative_slope = negative_slope
        self.norm = norm
        
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

        if conv_type == 'SAGEConv':
            modules.append((gnn.SAGEConv(input_dim, embed_dim), 'x, edge_index -> x'))
        elif conv_type == 'GATConv':
            modules.append((gnn.GATConv(input_dim, embed_dim), 'x, edge_index -> x'))
        elif conv_type == 'GCNConv':
            modules.append((gnn.GCNConv(input_dim, embed_dim), 'x, edge_index -> x'))
        else: 
            modules.append((gnn.SAGEConv(input_dim, embed_dim), 'x, edge_index -> x'))

        modules.append(nn.ReLU(inplace=True))

        modules.append((nn.Dropout(p = drop_rate), 'x -> x'))

        assert(np.log2(hidden_dim).is_integer())

        exp_layer_rate = np.log2(hidden_dim).astype(np.int32)


        # TODO: What if there is no hidden layer at all? 

        # first hidden layer to scale from embed dim to hidden dim

        input_dim_layer = embed_dim
        out_layer_dim = int(2 ** exp_layer_rate)

        # TODO: Do Upsampling and Downsampling again?? 
        
        for layer_idx in range(0, no_of_layers):
            
            modules.extend(self.get_hidden_layer(
                        input_dim = input_dim_layer, 
                        hidden_dim = out_layer_dim, 
                        norm = self.norm, 
                        drop_rate = self.drop_rate,
                        negative_slope = self.negative_slope
                        )
                    )
            
            input_dim_layer = out_layer_dim
            out_layer_dim = int(input_dim_layer * 2)


        # last input_dim_layer is the output layer of the last hidden layer
        modules.append(nn.Linear(input_dim_layer, class_num)) # no_of_layers * hidden_dim
        # modules.append(nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True))
        modules.append(nn.Softmax(-1))

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

    def get_hidden_layer(
            self, 
            input_dim = 16, 
            hidden_dim = 64, 
            conv_type = 'SAGEConv', 
            norm = 'Batch',
            drop_rate = 0.,
            negative_slope = 0.,):
        modules = []

        if conv_type == 'SAGEConv':
            modules.append((gnn.SAGEConv(input_dim, hidden_dim), 'x, edge_index -> x'))
        elif conv_type == 'GATConv':
            modules.append((gnn.GATConv(input_dim, hidden_dim), 'x, edge_index -> x'))
        elif conv_type == 'GCNConv':
            modules.append((gnn.GCNConv(input_dim, hidden_dim), 'x, edge_index -> x'))
        else: 
            modules.append((gnn.SAGEConv(input_dim, hidden_dim), 'x, edge_index -> x'))

        # Normalization before after convolution, but before activation
        # as it has been done in resnet as well (\cite He et al., Deep Residual Learning for Image Recognition)

        if norm == 'Batch':

            # most important assumption: samples should be i.i.d. -> basically never the case
            # However it still works
            # Use normalization to reduce covariance shift
            # layer normalization -> standardize each feature
            # layer norm is weaker -> but does the trick for covariance shift

            modules.append(gnn.BatchNorm(in_channels=hidden_dim, eps=1e-05, momentum=0.1))
        elif norm == 'Instance':    
            modules.append(gnn.InstanceNorm(in_channels=hidden_dim, eps=1e-05, momentum=0.1))

        modules.append(nn.ReLU(inplace=True))

        # Dropout
        modules.append((nn.Dropout(p=drop_rate), 'x -> x'))
        # modules.append(dropout_node())

        return modules
    # gnn.Sequential('x, edge_index',
    #                    modules)

    
    def forward(self, data):
        
#         x_embed = self.node_embeder(data['pos'], data['edge_index'])
        
        
        return self.node_embedder(data['x'], data['edge_index']) #self.mlp( self.aggr_1(x_embed, data['batch']) )