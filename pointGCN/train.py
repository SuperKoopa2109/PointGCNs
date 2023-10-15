# **** START Import Section

import argparse
import numpy as np
import random
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

from torch_geometric.utils import degree 

from tqdm import tqdm
import datetime

# For hyper parameter optimization
import optuna
import logging

from optuna.samplers import TPESampler

# For logging
import wandb

from sklearn.metrics import roc_curve, auc


from models.pointGCN import seg_model


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

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
    
class Config():
    def __init__(self, kwconf = None, **kwargs):
        if kwconf is not None:
            self.conf = {}
            for key, val in kwconf.items():
                setattr(self, key, val)
                
    def __getitem__(self, key):
        return getattr(self, key)



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


def log_string(out_str, log_file):
    log_file.write(out_str+'\n')
    log_file.flush()
    print(out_str)

def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def load_data(config: object, radius_threshold = 0.02):

    # TODO: add edge features to graph -> Distances ? Degree ? Nearest Neighbourhood -> argsort of distnaces and one-hot-encoded?
    
    # pre_transform = T.NormalizeScale()
    # TODO: Maybe use this tranform step?
    # transform = T.SamplePoints(config.sample_points)

    no_points_sampled = 2048

    # # Load train_dataset first to get first sample and determine OneHotDegree number
    # train_dataset = ShapeNet(
    #     root = config['savedir'] + "/" + config['model_name'],
    #     categories = config['categories'],
    #     transform=T.Compose([
    #                             T.FixedPoints(no_points_sampled,replace = False, allow_duplicates = True),
    #                             T.RadiusGraph(radius_threshold), # TODO: Maybe use k nearest neighbors? 
    #                             T.Distance()
    #                         ]), #T.OneHotDegree(50) just crashes if number too low  ## TODO: Check for highest degree on training data and use that
    #     split = "train"
    # )

    # # compute for in_degree = 0 (default)
    # # so degree for outgoing connections
    
    # # max degree of first sample
    # max_degree = 0
    # for i in range(5):
    #     new_degree = degree(train_dataset[0]['edge_index'][0]).max()
    #     if new_degree > max_degree:
    #         max_degree = new_degree
    
    # round to 100, with minimum of 100 ... It should be okay, if number is a little bit higher than actual degree.
    # Since we base the degree on only 5 samples, better to have some space to work with
    # max_degree = max_degree - max_degree % 100 if max_degree > 100 else 100

    max_degree = 100

    train_dataset = ShapeNet(
        root = config['savedir'] + "/" + config['model_name'],
        categories = config['categories'],
        pre_transform=T.Compose([
                                T.FixedPoints(no_points_sampled,replace = False, allow_duplicates = True),
                                T.RadiusGraph(radius_threshold), # TODO: Maybe use k nearest neighbors? 
                                T.Distance(),
                                T.OneHotDegree(max_degree) 
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
            pre_transform=T.Compose([
                                    T.FixedPoints(no_points_sampled,replace = False, allow_duplicates = True),
                                    T.RadiusGraph(radius_threshold),
                                    T.Distance(),
                                    T.OneHotDegree(max_degree) 
                                        ]),
            split = "val"
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    # random_indices = np.random.choice(range(len(val_dataset)), size = config.vis_sample_size, replace = False)
    vis_indices = [1,2,21]

    vis_loader = DataLoader(
        [val_dataset[idx] for idx in vis_indices],
        batch_size = 1,
        shuffle = False,
        num_workers = config['num_workers']
    )



    # vis_loader = DataLoader(
    #     vis_dataset, 
    #     batch_size = 
    # )

    test_dataset = ShapeNet(
        root = config['savedir'] + "/" + config['model_name'],
        categories = config['categories'],
        pre_transform=T.Compose([
                                    T.FixedPoints(no_points_sampled,replace = False, allow_duplicates = True), 
                                    T.RadiusGraph(radius_threshold),
                                    T.Distance(),
                                    T.OneHotDegree(max_degree) 
                                ]), #T.OneHotDegree(50) just crashes if number too low  ## TODO: Check for highest degree on training data and use that
        split = "test"
    )
        
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True
    #     num_workers=config['num_workers']
    )

    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, vis_loader

def load_model(
        modelname, 
        input_dim, 
        embed_dim=64, 
        hidden_dim=128,
        conv_type='SAGEConv',
        no_of_layers=3, 
        class_num=4,
        drop_rate = 0.,
        negative_slope = 0., 
        norm = 'None'
        ):
                
    if modelname == 'SageNet':
        return seg_model(
             input_dim=input_dim, 
             embed_dim=embed_dim, 
             hidden_dim=hidden_dim,
             conv_type=conv_type,
             no_of_layers=no_of_layers, 
             class_num=class_num,
             drop_rate = drop_rate,
             negative_slope=negative_slope,
             norm=norm
             )
    else:
        return seg_model(
             input_dim=input_dim, 
             embed_dim=embed_dim, 
             hidden_dim=hidden_dim, 
             conv_type=conv_type,
             no_of_layers=no_of_layers, 
             class_num=class_num,
             drop_rate = drop_rate,
             negative_slope=negative_slope,
             norm=norm
             )

def get_metric(y_pred, y_true, metric_type = 'accuracy'):
    
    metric_val = 0

    if metric_type == 'auc':
        fpr, tpr, thresh = roc_curve(y_pred, y_true)

        metric_val = auc(fpr, tpr)


    return metric_val




def objective(trial):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb_project = "pointGCNs"
    wandb_run_name = "playground/shapnet/1"

    modulename = 'wandb'
    if is_imported(modulename):
        wandb.init(
            project=wandb_project,
            #name=wandb_run_name, 
            job_type="baseline-train"
            )

        wandb_run_name = wandb.run.name

        # Set experiment configs to be synced with wandb
        config = wandb.config

    else:
        print(f'{modulename} not imported || Run will only be logged locally')

        config = Config({
            "model_name": "GSegNet",
            "categories": "Airplane",
        })

    config.seed = 42
    config.study_name = trial.study.study_name
    config.model_name = "GSegNet"
    config.categories = "Airplane"
    config.metric = FLAGS.metric
    config.savedir = "data"
    config.logdir = LOG_DIR
    config.radius_threshold = trial.suggest_float('radius_threshold', low=0.02, high=0.06, step=0.02, log=False)
    config.batch_size = trial.suggest_int('batch_size', low=16, high=32, step=16)
    config.num_workers = 1
    config.optimizer = "Adam" # Could be done in the future: trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
    config.epochs = trial.suggest_int('epoch_count', low=20, high=60, step=20)
    config.embed_dim=trial.suggest_int('embed_dim', low=64, high=128, step=64)
    config.hidden_layers = trial.suggest_int("num_layers", 2, 5)
    config.conv_layer = trial.suggest_categorical('conv_layer', ['SAGEConv', 'GATConv', 'GCNConv'])
    config.hidden_dim=trial.suggest_int('hidden_dim', low=128, high=256, step=128)
    config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config.drop_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
    config.negative_slope = trial.suggest_float("negative_slope", 0.0, 0.2)
    config.norm_layer = trial.suggest_categorical("norm_layer", ["None", "Batch", "Instance"])
    config.vis_sample_size = 3
    config.wandb_run_name = wandb_run_name
    config.use_drive_storage = True

    seed_everything(config.seed)

    table = wandb.Table(
            columns=[
                "Epoch",
                "Ground-Truth",
                "Predicted-Classes"
            ]
        )

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, vis_loader = load_data(config, radius_threshold = config.radius_threshold)

    sample = next(iter(train_dataset))

    model = load_model(
        modelname = config.model_name, 
        input_dim = sample['x'].shape[1], 
        embed_dim=config.embed_dim, 
        hidden_dim=config.hidden_dim,  #dimensions of first hidden layer
        conv_type=config.conv_layer,
        no_of_layers = config.hidden_layers,
        class_num=int(sample['y'].max() + 1),
        drop_rate = config.drop_rate,
        negative_slope = config.negative_slope, 
        norm = config.norm_layer
        ).to(device)
                

    print(model)

    # TODO: use .unique instead ?? 


    # Define Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate
    )

    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=config.learning_rate, momentum=0.9
    # )

    loss = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        train_loss, train_accuracy = train_step(epoch, model, optimizer, loss, train_loader, device, config)
        val_loss, val_accuracy = val_step(epoch, model, loss, val_loader, device, config)
        table = visualize_evaluation(epoch, model, table, vis_loader, config, device)

        trial.report(val_accuracy, epoch)
        
        # Handle pruning based on the validation accuracy.
        if trial.should_prune():
            if is_imported('wandb'):
                wandb.log({"PredClass_vs_TrueClass": table})
                wandb.finish()
            raise optuna.TrialPruned()

    res = test_model(model, loss, test_loader, device, config)

    if is_imported('wandb'):
            wandb.log({"PredClass_vs_TrueClass": table})
            wandb.finish()

    return val_accuracy #res['Test/Accuracy']


def train(FLAGS):

    if FLAGS.train_hyperparams == "True":

        if FLAGS.existing_study != 'None':
            study_name = FLAGS.existing_study    
        else:            
            now = datetime.datetime.now()
            date_time = now.strftime("%Y_%m_%d_%H_%M")

            study_name = "pointGCN_study_" + date_time


        if FLAGS.use_drive_storage == "True":
            # Add stream handler of stdout to show the messages
            optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            study_dir = os.path.join('drive', 'MyDrive', 'PointGCN', 'logs')
            study_path = os.path.join(study_dir, study_name)
            storage_name = "sqlite:///{}.db".format(study_path)
        else:
            storage_name = None

        study = optuna.create_study(
                study_name=study_name, 
                storage=storage_name, 
                load_if_exists=True,
                sampler=TPESampler(), # choice based on: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
                pruner=optuna.pruners.MedianPruner(),
                directions=["maximize"]
            )
        
        study.optimize(objective, n_trials=20)
    else:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        wandb_project = "pointGCNs"
        wandb_run_name = "playground/shapnet/1"

        modulename = 'wandb'
        if is_imported(modulename):
            wandb.init(
                project=wandb_project, 
                #name=wandb_run_name, 
                job_type="baseline-train"
                )

            wandb_run_name = wandb.run.name

            # Set experiment configs to be synced with wandb
            config = wandb.config

        else:
            print(f'{modulename} not imported || Run will only be logged locally')

            config = Config({
                "model_name": "GSegNet",
                "categories": "Airplane",
            })

        config.seed = 42
        config.model_name = "ShapeNet"
        config.categories = "Airplane"
        config.metric = FLAGS.metric
        config.savedir = "data"
        config.logdir = LOG_DIR
        config.batch_size = 32
        config.num_workers = 1
        config.optimizer = "Adam" # Could be done in the future: trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
        config.epochs = FLAGS.epochs
        config.embed_dim = 64
        config.hidden_layers = FLAGS.layers
        config.conv_layer = 'SAGEConv' #, 'GATConv', 'GCNConv'
        config.hidden_dim = 128
        config.learning_rate = 1e-5 #, 1e-2
        config.drop_rate = 0.0
        config.negative_slope = 0.0 #, 0.2
        config.norm_layer = 'Batch' # "None", "Batch", "Instance"
        config.vis_sample_size = 3
        config.wandb_run_name = wandb_run_name

        seed_everything(config.seed)

        table = wandb.Table(
            columns=[
                "Epoch",
                "Ground-Truth",
                "Predicted-Classes"
            ]
        )

        train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, vis_loader = load_data(config)

        sample = next(iter(train_dataset)) #train_loader.dataset[0] #['y']
        
        
        model = load_model(
            modelname = 'SageNet', 
            input_dim = sample['x'].shape[1], 
            embed_dim=config.embed_dim, 
            hidden_dim=config.hidden_dim,  #dimensions of first hidden layer
            conv_type=config.conv_layer,
            no_of_layers = config.hidden_layers,
            class_num=int(sample['y'].max() + 1),
            drop_rate = config.drop_rate,
            negative_slope = config.negative_slope, 
            norm = config.norm_layer
        ).to(device)

        # model = load_model(
        #     modelname = 'SageNet', 
        #     input_dim = sample['x'].shape[1], 
        #     no_of_layers = config.hidden_layers,
        #     hidden_dim=128, 
        #     embed_dim=64, 
        #     class_num=int(sample['y'].max() + 1)).to(device)
        print(model)

        # TODO: use .unique instead ?? 


        # Define Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )

        loss = nn.CrossEntropyLoss()

        for epoch in range(config.epochs):
            train_step(epoch, model, optimizer, loss, train_loader, device, config)
            val_step(epoch, model, loss, val_loader, device, config)
            visualize_evaluation(epoch, model, table, vis_loader, config, device)
        # load model

        

        if is_imported('wandb'):
            wandb.log({"PredClass_vs_TrueClass": table})
            wandb.finish()

def get_model(FLAGS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = Config({
        "model_name": "GSegNet",
        "categories": "Airplane",
    })

    config.seed = 42
    config.model_name = "ShapeNet"
    config.categories = "Airplane"
    config.metric = FLAGS.metric
    config.savedir = "data"
    config.logdir = LOG_DIR
    config.batch_size = 32
    config.num_workers = 1
    config.optimizer = "Adam" # Could be done in the future: trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
    config.epochs = FLAGS.epochs
    config.embed_dim = 64
    config.hidden_layers = FLAGS.layers
    config.conv_layer = 'SAGEConv' #, 'GATConv', 'GCNConv'
    config.hidden_dim = 128
    config.learning_rate = 1e-5 #, 1e-2
    config.drop_rate = 0.0
    config.negative_slope = 0.0 #, 0.2
    config.norm_layer = 'Batch' # "None", "Batch", "Instance"

    seed_everything(config.seed)

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, vis_loader = load_data(config)

    sample = next(iter(train_dataset))
    
    
    model = load_model(
        modelname = 'SageNet', 
        input_dim = sample['x'].shape[1], 
        embed_dim=config.embed_dim, 
        hidden_dim=config.hidden_dim,  #dimensions of first hidden layer
        conv_type=config.conv_layer,
        no_of_layers = config.hidden_layers,
        class_num=int(sample['y'].max() + 1),
        drop_rate = config.drop_rate,
        negative_slope = config.negative_slope, 
        norm = config.norm_layer
    ).to(device)

    return model


def train_step(epoch, model, optimizer, loss, train_loader, device, config):
    """Training Step"""
    model.train()
    epoch_loss, correct, total_predictions = 0, 0, 0
    num_train_examples = len(train_loader)
    
    if config.metric == 'auc':
        metric_name = "Train/AUC_Score"
        auc_scores = np.zeros(num_train_examples)
    else:
        metric_name = "Train/Accuracy"
    
    progress_bar = tqdm(
        range(num_train_examples),
        desc=f"Training Epoch {epoch + 1}/{config.epochs}"
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

        if config.metric == 'auc':
            auc_scores[batch_idx] = get_metric(class_pred, data['y'])
        else:
            correct += class_pred.eq(data['y']).sum().item()
            total_predictions += data['x'].shape[0]
    
    epoch_loss = epoch_loss / num_train_examples
    # epoch_accuracy = correct / total_predictions

    if config.metric == 'auc':
        epoch_metric = auc_scores.sum() / num_train_examples
    else: 
        epoch_metric = correct / total_predictions
    
    print(f'epoch_loss: {epoch_loss} \n epoch_accuracy {epoch_metric}')

    if is_imported('wandb'):
        wandb.log({
            "Train/Loss": epoch_loss,
            metric_name: epoch_metric
        })

    return (epoch_loss, epoch_metric)

# for epoch in range(config.epochs):
#     train_step(epoch)
    
#     wandb.log({
#         "Train/Loss": epoch_loss,
#         "Train/Accuracy": epoch_accuracy
#     })


def val_step(epoch, model, loss, val_loader, device, config):
    """Validation Step"""
    model.eval()
    epoch_loss, correct, total_predictions = 0, 0, 0
    num_val_examples = len(val_loader)

    if config.metric == 'auc':
        metric_name = "Validation/AUC_Score"
        auc_scores = np.zeros(num_val_examples)
    else:
        metric_name = "Validation/Accuracy"
    
    progress_bar = tqdm(
        range(num_val_examples),
        desc=f"Validation Epoch {epoch + 1}/{config.epochs}"
    )

    for batch_idx in progress_bar:
        data = next(iter(val_loader)).to(device)
        
        with torch.no_grad():
            prediction = model(data)
        
        l = loss(prediction, data['y'])
        # epoch_loss += l.item()
        # correct += prediction.max(1)[1].eq(data.y).sum().item()

        epoch_loss += l.item()
        class_pred = prediction.max(1)[1]

        if config.metric == 'auc':
            auc_scores[batch_idx] = get_metric(class_pred, data['y'])
        else:
            correct += class_pred.eq(data['y']).sum().item()
            total_predictions += data['x'].shape[0]

        # correct += class_pred.eq(data['y']).sum().item()
        # total_predictions += data['x'].shape[0]
    
    epoch_loss = epoch_loss / num_val_examples
    # epoch_accuracy = correct / total_predictions
        
    if config.metric == 'auc':
        epoch_metric = auc_scores.sum() / num_val_examples
    else: 
        epoch_metric = correct / total_predictions
            
    
    # epoch_loss = epoch_loss / num_val_examples
    # epoch_accuracy = correct / len(val_loader.dataset)
    
    print(f'*** VALIDATION ***')
    print(f'epoch_loss: {epoch_loss} \n {metric_name} {epoch_metric}')

    log = {
        "Validation/Loss": epoch_loss,
        metric_name: epoch_metric
    }

    if is_imported('wandb'):
        wandb.log(log)

    return (epoch_loss, epoch_metric)


def visualize_evaluation(epoch, model, table, vis_loader, config, device):
    """Visualize validation result in a Weights & Biases Table"""
    predictions, ground_truths = [], []
    progress_bar = tqdm(
        range(config.vis_sample_size),
        desc=f"Generating Visualizations for Epoch {epoch + 1}/{config.epochs}"
    )

    # # determine all shapes for data
    # data = next(iter(vis_loader)).to(device)

    


    vis_iter = iter(vis_loader)

    for idx in progress_bar:
        data = next(vis_iter).to(device)

        with torch.no_grad():
            logit_preds = model(data)

        if idx == 0:
            no_of_samples, embedded_dim = data['x'].size()

            all_logit_preds = torch.zeros([config.vis_sample_size, no_of_samples, logit_preds.size()[-1]])
            all_preds = torch.zeros([config.vis_sample_size, no_of_samples])
            all_trues = torch.zeros([config.vis_sample_size, no_of_samples])
            all_norm_data = torch.zeros([config.vis_sample_size, no_of_samples, embedded_dim])
            all_positions = torch.zeros([config.vis_sample_size, no_of_samples, data['pos'].size()[-1]])

        
        
        
        preds = logit_preds.max(1)[1]

        log_run_path = os.path.join(config.logdir, config.wandb_run_name)
        if not os.path.exists(log_run_path): os.mkdir(log_run_path)
        # os.mkdir(os.path.join(config.logdir, config.wandb_run_name))

        all_logit_preds[idx] = logit_preds
        all_preds[idx] = preds
        all_trues[idx] = data['y']
        all_norm_data[idx] = data['x']
        all_positions[idx] = data['pos']

        predictions.append(
            wandb.Object3D(torch.squeeze(torch.hstack([data['pos'], preds.reshape(-1, 1)]), dim=0).cpu().numpy())
        )
        ground_truths.append(
            wandb.Object3D(torch.squeeze(torch.hstack([data['pos'], data['y'].reshape(-1, 1)]), dim=0).cpu().numpy())
        )


    torch.save(all_logit_preds, os.path.join(log_run_path, f'logit_preds_{config.wandb_run_name}_epoch_{epoch}.pt'))
    torch.save(all_preds, os.path.join(log_run_path, f'preds_{config.wandb_run_name}_epoch_{epoch}.pt'))
    torch.save(all_trues, os.path.join(log_run_path, f'trues_{config.wandb_run_name}_epoch_{epoch}.pt'))
    torch.save(all_norm_data, os.path.join(log_run_path, f'norm_data_{config.wandb_run_name}_epoch_{epoch}.pt'))
    torch.save(all_positions, os.path.join(log_run_path, f'positions_{config.wandb_run_name}_epoch_{epoch}.pt'))

    # Store 3D models every 5 epochs
    if ((epoch + 1) % 5 == 0):
        table.add_data(
            epoch, ground_truths, predictions
        )
    return table

def test_model(model, loss, test_loader, device, config):
    """Testing Step"""
    model.eval()
    epoch_loss, correct, total_predictions = 0, 0, 0
    num_test_examples = len(test_loader)


    if config.metric == 'auc':
        metric_name = "Test/AUC_Score"
        auc_scores = np.zeros(num_test_examples)
    else:
        metric_name = "Test/Accuracy"
    

    progress_bar = tqdm(
        range(num_test_examples),
        desc=f"Testing model"
    )

    for batch_idx in progress_bar:
        data = next(iter(test_loader)).to(device)
        
        with torch.no_grad():
            prediction = model(data)
        
        l = loss(prediction, data['y'])
        # epoch_loss += l.item()
        # correct += prediction.max(1)[1].eq(data.y).sum().item()

        epoch_loss += l.item()
        class_pred = prediction.max(1)[1]

        if config.metric == 'auc':
            auc_scores[batch_idx] = get_metric(class_pred, data['y'])
        else:
            correct += class_pred.eq(data['y']).sum().item()
            total_predictions += data['x'].shape[0]

        # correct += class_pred.eq(data['y']).sum().item()
        # total_predictions += data['x'].shape[0]


    if config.metric == 'auc':
        # calc average auc score
        epoch_metric = auc_scores.sum() / num_test_examples
    else: 
        epoch_metric = correct / total_predictions

    epoch_loss = epoch_loss / num_test_examples

    # epoch_accuracy = correct / total_predictions
            
    
    print(f'*** TESTING ***')
    print(f'epoch_loss: {epoch_loss} \n {metric_name} {epoch_metric}')

    log = {
        "Test/Loss": epoch_loss,
        metric_name: epoch_metric
    }

    if is_imported('wandb'):
        wandb.log(log)

    return log


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
        parser.add_argument('--epochs', type=int, default=5, help='No of epochs for training [default: 5]')
        parser.add_argument('--layers', type=int, default=3, help='No of hidden layers [default: 3]')
        parser.add_argument('--dataset', default='shapenet', help='Dataset to be used for prediction [default: modelnet40]')
        parser.add_argument('--colab', default='False', help='Code is executed in Google colab [default: False]')
        parser.add_argument('--train_hyperparams', default='False', help='Hyper param tuning [default: False]')
        parser.add_argument('--use_drive_storage', default='False', help='Store results in connected Google Drive [default: False]')
        parser.add_argument('--existing_study', default='None', help='existing study can be passed to continue a started study that has not been finished [default: None]')
        parser.add_argument('--metric', default='accuracy', help='define metric to be used for evaluation \{auc, accuracy\} [default: accuracy]')
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

    LOG_DIR = FLAGS.log_dir
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    # os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
    if FLAGS.colab == 'True': #is_running_in_colab():
        os.system('cp PointGCNs/train.py %s' % (LOG_DIR)) # bkp of train procedure
    else:
        os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(FLAGS)+'\n')

    train(FLAGS=FLAGS)
    LOG_FOUT.close()
