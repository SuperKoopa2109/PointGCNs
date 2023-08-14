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

from tqdm import tqdm

# For hyper parameter optimization
import optuna
from optuna.samplers import TPESampler

# For logging
import wandb


from models.pointGCN import SAGE_model


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
    

def load_data(config: object):

    # TODO: add edge features to graph -> Distances ? Degree ? Nearest Neighbourhood -> argsort of distnaces and one-hot-encoded?
    
    # pre_transform = T.NormalizeScale()
    # TODO: Maybe use this tranform step?
    # transform = T.SamplePoints(config.sample_points)

    train_dataset = ShapeNet(
        root = config['savedir'] + "/" + config['model_name'],
        categories = config['categories'],
        transform=T.Compose([
                                T.FixedPoints(2048,replace = False, allow_duplicates = True),
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
            transform=T.Compose([
                                    T.FixedPoints(2048,replace = False, allow_duplicates = True),
                                    T.RadiusGraph(0.01),
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

    random_indices = np.random.choice(range(len(val_dataset)), size = config.vis_sample_size, replace = False)

    vis_loader = DataLoader(
        [val_dataset[idx] for idx in random_indices],
        batch_size = 1,
        shuffle = True,
        num_workers = config['num_workers']
    )



    # vis_loader = DataLoader(
    #     vis_dataset, 
    #     batch_size = 
    # )

    test_dataset = ShapeNet(
        root = config['savedir'] + "/" + config['model_name'] + "_test",
        categories = config['categories'],
        transform=T.Compose([
                                    T.FixedPoints(2048,replace = False, allow_duplicates = True), 
                                    T.RadiusGraph(0.01),
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

    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, vis_loader

def load_model(
        modelname, 
        input_dim, 
        embed_dim=64, 
        hidden_dim=128, 
        no_of_layers=3, 
        class_num=4,
        drop_rate = 0.,
        negative_slope = 0., 
        norm = 'None'
        ):
                
    if modelname == 'SageNet':
        return SAGE_model(
             input_dim=input_dim, 
             embed_dim=embed_dim, 
             hidden_dim=hidden_dim, 
             no_of_layers=no_of_layers, 
             class_num=class_num,
             drop_rate = drop_rate,
             negative_slope=negative_slope,
             norm=norm
             )
    else:
        return SAGE_model(
             input_dim=input_dim, 
             embed_dim=embed_dim, 
             hidden_dim=hidden_dim, 
             no_of_layers=no_of_layers, 
             class_num=class_num,
             drop_rate = drop_rate,
             negative_slope=negative_slope,
             norm=norm
             )

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
            "model_name": "ShapeNet",
            "categories": "Airplane",
        })

    config.seed = 42
    config.model_name = "ShapeNet"
    config.categories = "Airplane"
    config.savedir = "data"
    config.logdir = LOG_DIR
    config.batch_size = trial.suggest_int('epoch_count', low=32, high=128, step=32)
    config.num_workers = 1
    config.optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
    config.epochs = trial.suggest_int('epoch_count', low=50, high=200, step=50)
    config.embed_dim=trial.suggest_int('embed_dim', low=64, high=128, step=64)
    config.hidden_layers = trial.suggest_int("num_layers", 1, 4)
    config.hidden_dim=trial.suggest_int('hidden_dim', low=128, high=256, step=128)
    config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config.drop_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)
    config.negative_slope = trial.suggest_float("negative_slope", 0.0, 0.2)
    config.norm_layer = trial.suggest_categorical("norm_layer", ["None", "Batch", "Instance"])
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

    sample = next(iter(train_dataset))

    model = load_model(
        modelname = 'SageNet', 
        input_dim = sample['x'].shape[1], 
        embed_dim=trial.config.embed_dim, 
        hidden_dim=config.hidden_dim,  #dimensions of first hidden layer
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

    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=0.9
        )

    loss = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        train_step(epoch, model, optimizer, loss, train_loader, device, config)
        val_step(epoch, model, loss, val_loader, device, config)
        visualize_evaluation(epoch, model, table, vis_loader, config, device)


def train(FLAGS, hyperparam_tuning = False):

    if hyperparam_tuning == "True":
        study = optuna.create_study(sampler=TPESampler(), directions=["minimize"])
        #study.optimize(objective, n_trials=100, timeout=300)
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

        config.seed = 42
        config.model_name = "ShapeNet"
        config.categories = "Airplane"
        config.savedir = "data"
        config.logdir = LOG_DIR
        config.batch_size = 32
        config.num_workers = 1
        config.epochs = FLAGS.epochs
        config.hidden_layers = FLAGS.layers
        config.learning_rate = FLAGS.learning_rate
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
            no_of_layers = config.hidden_layers,
            hidden_dim=128, 
            embed_dim=64, 
            class_num=int(sample['y'].max() + 1)).to(device)
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



def train_step(epoch, model, optimizer, loss, train_loader, device, config):
    """Training Step"""
    model.train()
    epoch_loss, correct, total_predictions = 0, 0, 0
    num_train_examples = len(train_loader)
    
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
        correct += class_pred.eq(data['y']).sum().item()
        total_predictions += data['x'].shape[0]
    
    epoch_loss = epoch_loss / num_train_examples
    epoch_accuracy = correct / total_predictions
    
    print(f'epoch_loss: {epoch_loss} \n epoch_accuracy {epoch_accuracy}')

    if is_imported('wandb'):
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
    epoch_loss, correct, total_predictions = 0, 0, 0
    num_val_examples = len(val_loader)
    
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
        correct += class_pred.eq(data['y']).sum().item()
        total_predictions += data['x'].shape[0]
    
    epoch_loss = epoch_loss / num_val_examples
    epoch_accuracy = correct / total_predictions
        
        # if batch_idx < 6:
            
    
    # epoch_loss = epoch_loss / num_val_examples
    # epoch_accuracy = correct / len(val_loader.dataset)
    
    print(f'*** VALIDATION ***')
    print(f'epoch_loss: {epoch_loss} \n epoch_accuracy {epoch_accuracy}')

    log = {
        "Validation/Loss": epoch_loss,
        "Validation/Accuracy": epoch_accuracy
    }

    if is_imported('wandb'):
        wandb.log(log)

    return log

def visualize_evaluation(epoch, model, table, vis_loader, config, device):
    """Visualize validation result in a Weights & Biases Table"""
    predictions, ground_truths = [], []
    progress_bar = tqdm(
        range(config.vis_sample_size),
        desc=f"Generating Visualizations for Epoch {epoch}/{config.epochs}"
    )

    # determine all shapes for data
    data = next(iter(vis_loader)).to(device)

    no_of_samples, embedded_dim = data['x'].size()


    with torch.no_grad():
            logit_preds = model(data)

    all_logit_preds = torch.zeros([config.vis_sample_size, no_of_samples, logit_preds.size()[-1]])
    all_preds = torch.zeros([config.vis_sample_size, no_of_samples])
    all_trues = torch.zeros([config.vis_sample_size, no_of_samples])
    all_norm_data = torch.zeros([config.vis_sample_size, no_of_samples, embedded_dim])
    all_positions = torch.zeros([config.vis_sample_size, no_of_samples, data['pos'].size()[-1]])

    for idx in progress_bar:
        data = next(iter(vis_loader)).to(device)
        
        with torch.no_grad():
            logit_preds = model(data)
        
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

    train(FLAGS=FLAGS, hyperparam_tuning=True)
    LOG_FOUT.close()
