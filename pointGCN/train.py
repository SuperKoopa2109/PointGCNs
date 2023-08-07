# **** START Import Section

import argparse
import numpy as np
# import tensorflow as tf
import socket
import importlib
import os
import sys

# append parent directory to system paths

import torch
from torch import nn

import torch_geometric.transforms as T

# from torch_geometric.datasets import ShapeNet #ModelNet

from dataset.shapenet_loader import ShapeNet

from torch_geometric.loader import DataLoader

from tqdm import tqdm

# For hyper parameter optimization
import optuna

# For logging
import wandb


from models.pointGCN import SAGE_model


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)

sys.path.append(BASE_DIR)

from param_config import param_config

def is_running_in_jupyter():
    try:
        # Check if the 'get_ipython' function exists
        shell = get_ipython().__class__.__name__

        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter Notebook or JupyterLab
        else:
            return False  # Other interactive shell
    except NameError:
        return False  # Not in an interactive shell

def is_running_in_colab():
    try:
        # Check if the 'get_ipython' function exists

        RunningInCOLAB = 'google.colab' in str(get_ipython())

        if RunningInCOLAB:
            return True
        else:
            return False  # Other interactive shell
    except NameError:
        return False  # Not in an interactive shell

def is_imported(modulename):
    return modulename in sys.modules

class Custom_Parser():
    gpu = 0
    model = 'pointnet_cls'
    log_dir = 'log'
    num_point = 1024
    max_epoch = 2
    batch_size = 32
    learning_rate = 0.001
    momentum = 0.9
    optimizer = 'adam'
    decay_step = 200000
    decay_rate = 0.7
    dataset = 'modelnet40'

    def __init__(self, dataset = 'modelnet40', colab = 'False'):
        self.colab = colab
        self.dataset = dataset

    # def parse_args(self):
    #     return {
    #         gpu: self.gpu,
    #         model: self.model,
    #         log_dir: self.log_dir,
    #         num_point: self.num_point,
    #         max_epoch: self.max_epoch,
    #         batch_size: self.batch_size,
    #         learning_rate: self.learning_rate,
    #         momentum: self.momentum,
    #         optimizer: self.optimizer,
    #         decay_step: self.decay_step,
    #         decay_rate: self.decay_rate,
    #         colab: self.colab
    #         dataset: self.dataset
    #     }
    
    def parse_args(self):
        return self

# differentiate between running in an interactive shell vs shell
RunningInCOLAB = False
if is_running_in_jupyter():
    print("*** Code is running in an interactive Shell. ***")

    parser = Custom_Parser()

elif is_running_in_colab():
    print("*** Code is running in Google Colab. ***")

    RunningInCOLAB = True
    # BASE_DIR = os.path.join(BASE_DIR, 'pointGCNs')

    parser = Custom_Parser(colab = True)

else:
    print("*** Code is running in a Shell. ***")

    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    # parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    # parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
    # parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
    # parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    # parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    # parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    # parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
    # parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--dataset', default='shapenet', help='Dataset to be used for prediction [default: modelnet40]')
    parser.add_argument('--colab', default='False', help='Code is executed in Google colab')
    # FLAGS = parser.parse_args()

FLAGS = parser.parse_args()

# Check if code is executed in colab environment
# if FLAGS.colab == 'True':
    # BASE_DIR = os.path.join(BASE_DIR, 'PointGCNs')

param_config.set_value('paths', 'BASE_DIR', BASE_DIR)
param_config.set_value('paths', 'REPO_NAME', 'PointGCNs')
param_config.set_value('system', 'dataset', FLAGS.dataset)
param_config.set_value('system', 'RunningInCOLAB', FLAGS.colab)
param_config.save()

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

# **** END Import Section

# BATCH_SIZE = FLAGS.batch_size
# NUM_POINT = FLAGS.num_point
# MAX_EPOCH = FLAGS.max_epoch
# BASE_LEARNING_RATE = FLAGS.learning_rate
# GPU_INDEX = FLAGS.gpu
# MOMENTUM = FLAGS.momentum
# OPTIMIZER = FLAGS.optimizer
# DECAY_STEP = FLAGS.decay_step
# DECAY_RATE = FLAGS.decay_rate

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
# os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
if FLAGS.colab == 'True': #is_running_in_colab():
    os.system('cp PointGCNs/train.py %s' % (LOG_DIR)) # bkp of train procedure
else:
    os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def load_data(config):

    # TODO: add edge features to graph -> Distances ? Degree ? Nearest Neighbourhood -> argsort of distnaces and one-hot-encoded?
    
    # pre_transform = T.NormalizeScale()
    # TODO: Maybe use this tranform step?
    # transform = T.SamplePoints(config.sample_points)

    train_dataset = ShapeNet(
        root = config['savedir'] + "/" + config['model_name'],
        categories = config['categories'],
        transform=T.Compose([
                                #T.FixedPoints(2048,replace = False, allow_duplicates = False),
                                T.RadiusGraph(0.01),
                                T.Distance(),
                                T.OneHotDegree(50) 
                            ]), #T.OneHotDegree(50) just crashes if number too low  ## TODO: Check for highest degree on training data and use that
        split = "train"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )


    val_dataset = ShapeNet(
            root = config['savedir'] + "/" + config['model_name'],
            categories = config['categories'],
            transform=T.Compose([T.RadiusGraph(0.01),
                                        T.Distance(),
                                        T.OneHotDegree(50) 
                                        ]),
            split = "val"
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )


    test_dataset = ShapeNet(
        root = config['savedir'] + "/" + config['model_name'] + "_test",
        categories = config['categories'],
        transform=T.Compose([T.RadiusGraph(0.01),
                                      T.Distance(),
                                      T.OneHotDegree(50) 
                                      ]), #T.OneHotDegree(50) just crashes if number too low  ## TODO: Check for highest degree on training data and use that
        split = "test"
    )
        
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True
    #     num_workers=config['num_workers']
    )

    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader

def load_model(modelname = 'SageNet', input_dim=1, hidden_dim=2, embed_dim=3, class_num=6):
    if modelname == 'SageNet':
        return SAGE_model(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=embed_dim, class_num=class_num)
    else:
        return SAGE_model(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=embed_dim, class_num=class_num)

def train():

    wandb_project = "pointGCNs"
    wandb_run_name = "playground/shapnet/1"

    modulename = 'wandb'
    if is_imported(modulename):
        wandb.init(
            project=wandb_project, 
            name=wandb_run_name, 
            job_type="baseline-train"
            )

        # Set experiment configs to be synced with wandb
        config = wandb.config

    else:
        print(f'{modulename} not imported || Run will only be logged locally')


    

        class Config():
            def __init__(self, kwconf = None, **kwargs):
                if kwconf is not None:
                    self.conf = {}
                    for key, val in kwconf.items():
                        setattr(self, key, val)
                        
            def __getitem__(self, key):
                return getattr(self, key)

        config = Config({
            "model_name": "ShapeNet",
            "categories": "Airplane",
            "savedir": "data",
            "batch_size": 32,
            "num_workers": 1,
            "epochs": 50,
            "learning_rate": FLAGS.learning_rate
        })

    config.seed = 42
    config.model_name = "ShapeNet"
    config.categories = "Airplane"
    config.savedir = "data"
    config.batch_size = 32
    config.num_workers = 1
    config.epochs = 50
    config.learning_rate = FLAGS.learning_rate

    device = torch.device('cpu')
    
    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = load_data(config)


    # Define PointNet++ model.

    sample = next(iter(train_loader)) #train_loader.dataset[0] #['y']

    study = optuna.create_study()

    model = load_model(
        modelname = 'SageNet', 
        input_dim = sample['x'].shape[1], 
        hidden_dim=64, 
        embed_dim=128, 
        class_num=int(sample['y'].max() + 1)).to(device)
    print(model)

    # model = SAGE_model(
         
    #     128, 
    #     256, 
    #     class_num=int(sample['y'].max() + 1)).to(device)

    # TODO: use .unique instead ?? 


    # Define Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate
    )

    loss = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        train_step(epoch, model, optimizer, loss, train_loader, device, config)
        val_step(epoch, model, optimizer, loss, val_loader, device, config)
    # load model

    if is_imported('wandb'):
        wandb.finish()



def train_step(epoch, model, optimizer, loss, train_loader, device, config):
    """Training Step"""
    model.train()
    epoch_loss, correct, total_predictions = 0, 0, 0
    num_train_examples = len(train_loader)
    
    progress_bar = tqdm(
        range(num_train_examples),
        desc=f"Training Epoch {epoch}/{config.epochs}"
    )
    for batch_idx in progress_bar:
        data = next(iter(train_loader)).to(device)
        
        optimizer.zero_grad()
        prediction = model(data)
        l = loss(prediction, data['y']) #F.nll_loss(prediction, data.y) # TODO: Cross Entropy loss instead? 
        l.backward()
        optimizer.step()
        
        epoch_loss += l.item()
        class_pred = prediction.max(1)[1]
        correct += class_pred.eq(data['y']).sum().item()
        total_predictions += data['x'].shape[0]
    
    epoch_loss = epoch_loss / num_train_examples
    epoch_accuracy = correct / total_predictions
    
    print(f'epoch_loss: {epoch_loss} \n epoch_accuracy {epoch_accuracy}')

    wandb.log({
        "Train/Loss": epoch_loss,
        "Train/Accuracy": epoch_accuracy
    })

# for epoch in range(config.epochs):
#     train_step(epoch)
    
#     wandb.log({
#         "Train/Loss": epoch_loss,
#         "Train/Accuracy": epoch_accuracy
#     })


def val_step(epoch, model, loss, val_loader, device, config):
    """Validation Step"""
    model.eval()
    epoch_loss, correct = 0, 0
    num_val_examples = len(val_loader)
    
    progress_bar = tqdm(
        range(num_val_examples),
        desc=f"Validation Epoch {epoch}/{config.epochs}"
    )
    for batch_idx in progress_bar:
        data = next(iter(val_loader)).to(device)
        
        with torch.no_grad():
            prediction = model(data)
        
        l = loss(prediction, data.y)
        epoch_loss += l.item()
        correct += prediction.max(1)[1].eq(data.y).sum().item()
        
        # if batch_idx < 6:
            
    
    epoch_loss = epoch_loss / num_val_examples
    epoch_accuracy = correct / len(val_loader.dataset)
    
    log = {
        "Validation/Loss": epoch_loss,
        "Validation/Accuracy": epoch_accuracy
    }
    # wandb.log({
    #     "Validation/Loss": epoch_loss,
    #     "Validation/Accuracy": epoch_accuracy
    # })

    return log

# def visualize_evaluation(table, epoch):
#     """Visualize validation result in a Weights & Biases Table"""
#     point_clouds, losses, predictions, ground_truths, is_correct = [], [], [], [], []
#     progress_bar = tqdm(
#         range(config.num_visualization_samples),
#         desc=f"Generating Visualizations for Epoch {epoch}/{config.epochs}"
#     )
    
#     for idx in progress_bar:
#         data = next(iter(vizualization_loader)).to(device)
        
#         with torch.no_grad():
#             prediction = model(data)
        
#         point_clouds.append(
#             wandb.Object3D(torch.squeeze(data.pos, dim=0).cpu().numpy())
#         )
#         losses.append(F.nll_loss(prediction, data.y).item())
#         predictions.append(config.categories[int(prediction.max(1)[1].item())])
#         ground_truths.append(config.categories[int(data.y.item())])
#         is_correct.append(prediction.max(1)[1].eq(data.y).sum().item())
    
#     table.add_data(
#         epoch, point_clouds, losses, predictions, ground_truths, is_correct
#     )
#     return table


# def save_checkpoint(epoch):
#     """Save model checkpoints as Weights & Biases artifacts"""
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#     }, "checkpoint.pt")
    
#     artifact_name = wandb.util.make_artifact_name_safe(
#         f"{wandb.run.name}-{wandb.run.id}-checkpoint"
#     )
    
#     checkpoint_artifact = wandb.Artifact(artifact_name, type="checkpoint")
#     checkpoint_artifact.add_file("checkpoint.pt")
#     wandb.log_artifact(
#         checkpoint_artifact, aliases=["latest", f"epoch-{epoch}"]
#     )


if __name__ == "__main__":

    train()
    LOG_FOUT.close()
