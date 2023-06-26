# IMPORTS 

# import torch
# import torch_geometric
from torch import nn
# from torch_geometric import nn as gnn
# from torch_geometric import transforms
# from torch.nn import functional as F

# import numpy as np

# import matplotlib.pyplot as plt

# from tqdm import tqdm



# MODELS

class generic_class():
    def __init__(self, something):
        self.sth = something

    def read(self):
        return self.sth

class generic_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, class_num=6):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        
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

# class SAGE_model(nn.Module):
#     def __init__(self, input_dim, hidden_dim, embed_dim, class_num=6):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.embed_dim = embed_dim
#         self.hidden_dim = hidden_dim
#         self.class_num = class_num
        
# # TODO: *** ACTIVATION FUNCTIONS
# # TODO: for first activation function maybe only leaky ReLU makes sense
# # TODO: Hyperparameters for non-linear activation function -> possibly for final sigmoid or tanh
# # Tune on validation data

# # TODO: Hyper Param: Type of Conv Layer -> GAT / SAGE / GIN

# # TODO: Hyper Param: No of Conv Layer -> Repeating

# # TODO: add stochastic regularization -> Dropout?? Batch/Graph-Normalization?? 

# # TODO: Decide which target the hyper param optimization is supposed to optimise

# # TODO: Evaluation step: maybe use better metrics than accuracy
        
#         self.node_embeder = gnn.Sequential(
#             'x, edge_index',
#             [
#                 (gnn.SAGEConv(input_dim, hidden_dim), 'x, edge_index -> x'),
#                 nn.ReLU(inplace = True), # leaky ReLU instead?? 
#                 (gnn.SAGEConv(hidden_dim, embed_dim), 'x, edge_index -> x'),
#                 nn.ReLU(inplace = True),
#                 nn.Linear(embed_dim, class_num)
#                 #nn.ReLU(inplace = True)
#             ]
#         )
        
# #         self.aggr_1 = gnn.aggr.MeanAggregation()
# #         self.mlp = nn.Sequential(
# #             nn.Linear(embed_dim, class_num)
# #         )
        
        
        
# #         self.graph_embeder = gnn.Sequential(
# #         'x, edge_index, batch',
# #             [

# #             ]
# #         )
    
#     def forward(self, data):
        
# #         x_embed = self.node_embeder(data['pos'], data['edge_index'])
        
        
#         return self.node_embeder(data['pos'], data['edge_index']) #self.mlp( self.aggr_1(x_embed, data['batch']) )