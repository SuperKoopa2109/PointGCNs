import argparse
import subprocess
# import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os
import sys

import torch
import random

import wandb

# TMP IMPORT!!
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

from param_config import param_config
param_config.set_value('paths', 'BASE_DIR', BASE_DIR)
param_config.set_value('paths', 'REPO_NAME', 'PointGCNs')





# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--epoch', type=int, default=200, help='Epoch to run [default: 50]')
parser.add_argument('--point_num', type=int, default=2048, help='Point Number [256/512/1024/2048]')
parser.add_argument('--output_dir', type=str, default='train_results', help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--dataset', default='modelnet40', help='Dataset to be used for prediction [default: modelnet40]')
parser.add_argument('--colab', default='False', help='Code is executed in Google colab')
parser.add_argument('--use_drive', default='False', help='results are stored in google drive [default: False]')
parser.add_argument('--use_wandb', default='False', help='results are stored in weights and biases [default: False]')
parser.add_argument('--random_seed', type=int, default=42, help='random seed used [default: 42]')
FLAGS = parser.parse_args()

print(f"*******BASE_DIR: {param_config.get_value('paths', 'BASE_DIR')}*******")
param_config.set_value('system', 'dataset', FLAGS.dataset)
param_config.set_value('system', 'RunningInCOLAB', FLAGS.colab)
param_config.set_value('system', 'use_drive', FLAGS.use_drive)
param_config.set_value('system', 'use_wandb', FLAGS.use_wandb)
param_config.set_value('config', 'batchsize', str(FLAGS.batch))
param_config.save()

import provider
import pointnet_part_seg as model

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Enable TensorFlow 1 compatibility mode

import torch_geometric.transforms as T
# from torch_geometric.datasets import ShapeNet #ModelNet
# from torch_geometric.loader import DataLoader

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

seed_everything(FLAGS.random_seed)

hdf5_data_dir = os.path.join(BASE_DIR, 'hdf5_data')

# MAIN SCRIPT
point_num = FLAGS.point_num
batch_size = FLAGS.batch
output_dir = FLAGS.output_dir

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


if param_config.get_value('system', 'dataset') != 'shapenet':
    color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
    color_map = json.load(open(color_map_file, 'r'))

    all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
    fin = open(all_obj_cats_file, 'r')
    lines = [line.rstrip() for line in fin.readlines()]
    all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
    fin.close()

    all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
    NUM_CATEGORIES = 1 #16  ## JUST TRAINING FOR AIRPLANE!!! 
    NUM_PART_CATS = len(all_cats)
else:
    color_map_file = os.path.join(BASE_DIR, 'util', 'part_color_mapping.json')
    color_map = json.load(open(color_map_file, 'r'))



    NUM_CATEGORIES = 1 #16  ## JUST TRAINING FOR AIRPLANE!!! 
    NUM_PART_CATS = 4 # only four parts for airplane !!!
    all_obj_cats = [('Airplane', '02691156')] # only for airplane

print('#### Batch Size: {0}'.format(batch_size))
print('#### Point Number: {0}'.format(point_num))
print('#### Training using GPU: {0}'.format(FLAGS.gpu))

DECAY_STEP = 16881 * 20
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-5

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.001
MOMENTUM = 0.9
TRAINING_EPOCHES = FLAGS.epoch
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

if not FLAGS.colab:
    TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
    TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'val_hdf5_file_list.txt')

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER =  os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)

def printout(flog, data):
	print(data)
	flog.write(data + '\n')

def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))
    input_label_ph = tf.placeholder(tf.float32, shape=(batch_size, NUM_CATEGORIES))
    labels_ph = tf.placeholder(tf.int32, shape=(batch_size))
    seg_ph = tf.placeholder(tf.int32, shape=(batch_size, point_num))
    return pointclouds_ph, input_label_ph, labels_ph, seg_ph

def convert_label_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_ph, input_label_ph, labels_ph, seg_ph = placeholder_inputs()
            is_training_ph = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                            BASE_LEARNING_RATE,     # base learning rate
                            batch * batch_size,     # global_var indicating the number of steps
                            DECAY_STEP,             # step size
                            DECAY_RATE,             # decay rate
                            staircase=True          # Stair-case or continuous decreasing
                            )
            learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)
        
            bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
            bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

            lr_op = tf.summary.scalar('learning_rate', learning_rate)
            batch_op = tf.summary.scalar('batch_number', batch)
            bn_decay_op = tf.summary.scalar('bn_decay', bn_decay)
 
            labels_pred, seg_pred, end_points = model.get_model(pointclouds_ph, input_label_ph, \
                    is_training=is_training_ph, bn_decay=bn_decay, cat_num=NUM_CATEGORIES, \
                    part_num=NUM_PART_CATS, batch_size=batch_size, num_point=point_num, weight_decay=FLAGS.wd)

            # model.py defines both classification net and segmentation net, which share the common global feature extractor network.
            # In model.get_loss, we define the total loss to be weighted sum of the classification and segmentation losses.
            # Here, we only train for segmentation network. Thus, we set weight to be 1.0.
            loss, label_loss, per_instance_label_loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res  \
                = model.get_loss(labels_pred, seg_pred, labels_ph, seg_ph, 1.0, end_points)

            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            total_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            label_training_loss_ph = tf.placeholder(tf.float32, shape=())
            label_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            seg_training_loss_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            label_training_acc_ph = tf.placeholder(tf.float32, shape=())
            label_testing_acc_ph = tf.placeholder(tf.float32, shape=())
            label_testing_acc_avg_cat_ph = tf.placeholder(tf.float32, shape=())

            seg_training_acc_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_acc_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_acc_avg_cat_ph = tf.placeholder(tf.float32, shape=())

            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            total_test_loss_sum_op = tf.summary.scalar('total_testing_loss', total_testing_loss_ph)

            label_train_loss_sum_op = tf.summary.scalar('label_training_loss', label_training_loss_ph)
            label_test_loss_sum_op = tf.summary.scalar('label_testing_loss', label_testing_loss_ph)

            seg_train_loss_sum_op = tf.summary.scalar('seg_training_loss', seg_training_loss_ph)
            seg_test_loss_sum_op = tf.summary.scalar('seg_testing_loss', seg_testing_loss_ph)

            label_train_acc_sum_op = tf.summary.scalar('label_training_acc', label_training_acc_ph)
            label_test_acc_sum_op = tf.summary.scalar('label_testing_acc', label_testing_acc_ph)
            label_test_acc_avg_cat_op = tf.summary.scalar('label_testing_acc_avg_cat', label_testing_acc_avg_cat_ph)

            seg_train_acc_sum_op = tf.summary.scalar('seg_training_acc', seg_training_acc_ph)
            seg_test_acc_sum_op = tf.summary.scalar('seg_testing_acc', seg_testing_acc_ph)
            seg_test_acc_avg_cat_op = tf.summary.scalar('seg_testing_acc_avg_cat', seg_testing_acc_avg_cat_ph)

            train_variables = tf.trainable_variables()

            trainer = tf.train.AdamOptimizer(learning_rate)
            train_op = trainer.minimize(loss, var_list=train_variables, global_step=batch)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        
        init = tf.global_variables_initializer()
        sess.run(init)

        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')


        if param_config.get_value('system', 'dataset') == 'modelnet40': #FLAGS.dataset == 'modelnet40':
            train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
            num_train_file = len(train_file_list)
            test_file_list = provider.getDataFiles(TESTING_FILE_LIST)
            num_test_file = len(test_file_list)
        elif param_config.get_value('system', 'dataset') == 'shapenet': #FLAGS.dataset == 'shapenet':
            BATCH_NUM = 1
            num_train_file = BATCH_NUM
            num_test_file = BATCH_NUM

        fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        # write logs to the disk
        flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')

        if param_config.get_value('system', 'use_wandb') == 'True':

            wandb_project = "pointGCNs"
            wandb_run_name = "playground/shapnet/1"

            wandb.init(
                project=wandb_project,
                #name=wandb_run_name, 
                job_type="baseline-train"
                )

            wandb_run_name = wandb.run.name

            # Set experiment configs to be synced with wandb
            wandb_config = wandb.config

            wandb_config.batch_size = 512

            table = wandb.Table(
                columns=[
                    "Epoch",
                    "Ground-Truth",
                    "Predicted-Classes"
                ]
            )

        def train_one_epoch(train_file_idx, epoch_num):
            is_training = True

            # TODO: get file len; think of what should be in one batch/"train_file" and load that from dataset
            for i in range(num_train_file):
                
                if param_config.get_value('system', 'dataset') == 'shapenet': #FLAGS.dataset == 'shapenet':
                    cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(512, start_idx = i * 512)
                else:
                    cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[train_file_idx[i]])
                    printout(flog, 'Loading train file ' + cur_train_filename)
                    cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_train_filename)

                # cur_data, cur_labels, order = provider.shuffle_data(cur_data, np.squeeze(cur_labels))
                # cur_seg = cur_seg[order, ...]

                cur_labels_one_hot = convert_label_to_one_hot(cur_labels)

                num_data = len(cur_labels)
                num_batch = num_data // batch_size

                total_loss = 0.0
                total_label_loss = 0.0
                total_seg_loss = 0.0
                total_label_acc = 0.0
                total_seg_acc = 0.0

                for j in range(num_batch):
                    begidx = j * batch_size
                    endidx = (j + 1) * batch_size

                    feed_dict = {
                            pointclouds_ph: cur_data[begidx: endidx, ...], 
                            labels_ph: cur_labels[begidx: endidx, ...], 
                            input_label_ph: cur_labels_one_hot[begidx: endidx, ...], 
                            seg_ph: cur_seg[begidx: endidx, ...],
                            is_training_ph: is_training, 
                            }

                    _, loss_val, label_loss_val, seg_loss_val, per_instance_label_loss_val, \
                            per_instance_seg_loss_val, label_pred_val, seg_pred_val, pred_seg_res \
                            = sess.run([train_op, loss, label_loss, seg_loss, per_instance_label_loss, \
                            per_instance_seg_loss, labels_pred, seg_pred, per_instance_seg_pred_res], \
                            feed_dict=feed_dict)

                    per_instance_part_acc = np.mean(pred_seg_res == cur_seg[begidx: endidx, ...], axis=1)
                    average_part_acc = np.mean(per_instance_part_acc)

                    total_loss += loss_val
                    total_label_loss += label_loss_val
                    total_seg_loss += seg_loss_val
                    
                    per_instance_label_pred = np.argmax(label_pred_val, axis=1)
                    total_label_acc += np.mean(np.float32(per_instance_label_pred == cur_labels[begidx: endidx, ...]))
                    total_seg_acc += average_part_acc

                total_loss = total_loss * 1.0 / num_batch
                total_label_loss = total_label_loss * 1.0 / num_batch
                total_seg_loss = total_seg_loss * 1.0 / num_batch
                total_label_acc = total_label_acc * 1.0 / num_batch
                total_seg_acc = total_seg_acc * 1.0 / num_batch

                lr_sum, bn_decay_sum, batch_sum, train_loss_sum, train_label_acc_sum, \
                        train_label_loss_sum, train_seg_loss_sum, train_seg_acc_sum = sess.run(\
                        [lr_op, bn_decay_op, batch_op, total_train_loss_sum_op, label_train_acc_sum_op, \
                        label_train_loss_sum_op, seg_train_loss_sum_op, seg_train_acc_sum_op], \
                        feed_dict={total_training_loss_ph: total_loss, label_training_loss_ph: total_label_loss, \
                        seg_training_loss_ph: total_seg_loss, label_training_acc_ph: total_label_acc, \
                        seg_training_acc_ph: total_seg_acc})

                train_writer.add_summary(train_loss_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(train_label_loss_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(train_seg_loss_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(lr_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(bn_decay_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(train_label_acc_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(train_seg_acc_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(batch_sum, i + epoch_num * num_train_file)

                printout(flog, '\tTraining Total Mean_loss: %f' % total_loss)
                printout(flog, '\t\tTraining Label Mean_loss: %f' % total_label_loss)
                printout(flog, '\t\tTraining Label Accuracy: %f' % total_label_acc)
                printout(flog, '\t\tTraining Seg Mean_loss: %f' % total_seg_loss)
                printout(flog, '\t\tTraining Seg Accuracy: %f' % total_seg_acc)

                return (total_loss, total_label_loss, total_label_acc, total_seg_loss, total_seg_acc)

        def eval_one_epoch(epoch_num):
            is_training = False

            total_loss = 0.0
            total_label_loss = 0.0
            total_seg_loss = 0.0
            total_label_acc = 0.0
            total_seg_acc = 0.0
            total_seen = 0

            total_label_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
            total_seg_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
            total_seen_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.int32)

            for i in range(num_test_file):

                if param_config.get_value('system', 'dataset') == 'shapenet': #FLAGS.dataset == 'shapenet':
                    cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(256, is_training=False, start_idx = i * 256)
                else:
                    cur_test_filename = os.path.join(hdf5_data_dir, test_file_list[i])
                    printout(flog, 'Loading test file ' + cur_test_filename)

                    cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_test_filename)

                cur_labels = np.squeeze(cur_labels)

                cur_labels_one_hot = convert_label_to_one_hot(cur_labels)

                # print(f'cur_data.shape {cur_data.shape}')
                # print(f'cur_labels.shape {cur_labels.shape}')
                # print(f'cur_labels_one_hot.shape {cur_labels_one_hot.shape}')
                # print(f'seg_ph.shape {seg_ph.shape}')

                num_data = len(cur_labels)
                num_batch = num_data // batch_size

                for j in range(num_batch):
                    begidx = j * batch_size
                    endidx = (j + 1) * batch_size
                    feed_dict = {
                            pointclouds_ph: cur_data[begidx: endidx, ...], 
                            labels_ph: cur_labels[begidx: endidx, ...], 
                            input_label_ph: cur_labels_one_hot[begidx: endidx, ...], 
                            seg_ph: cur_seg[begidx: endidx, ...],
                            is_training_ph: is_training, 
                            }

                    # labels = ['labels_ph', 'input_label_ph', 'seg_ph']
                    # for idx, val in enumerate(feed_dict.values()):
                    #     if idx > 0 and idx < 4:
                    #         print(labels[idx - 1], val)

                    loss_val, label_loss_val, seg_loss_val, per_instance_label_loss_val, \
                            per_instance_seg_loss_val, label_pred_val, seg_pred_val, pred_seg_res \
                            = sess.run([loss, label_loss, seg_loss, per_instance_label_loss, \
                            per_instance_seg_loss, labels_pred, seg_pred, per_instance_seg_pred_res], \
                            feed_dict=feed_dict)

                    per_instance_part_acc = np.mean(pred_seg_res == cur_seg[begidx: endidx, ...], axis=1)
                    average_part_acc = np.mean(per_instance_part_acc)

                    total_seen += 1
                    total_loss += loss_val
                    total_label_loss += label_loss_val
                    total_seg_loss += seg_loss_val
                    
                    per_instance_label_pred = np.argmax(label_pred_val, axis=1)
                    total_label_acc += np.mean(np.float32(per_instance_label_pred == cur_labels[begidx: endidx, ...]))
                    total_seg_acc += average_part_acc

                    for shape_idx in range(begidx, endidx):
                        total_seen_per_cat[cur_labels[shape_idx]] += 1
                        total_label_acc_per_cat[cur_labels[shape_idx]] += np.int32(per_instance_label_pred[shape_idx-begidx] == cur_labels[shape_idx])
                        total_seg_acc_per_cat[cur_labels[shape_idx]] += per_instance_part_acc[shape_idx - begidx]

            total_loss = total_loss * 1.0 / total_seen
            total_label_loss = total_label_loss * 1.0 / total_seen
            total_seg_loss = total_seg_loss * 1.0 / total_seen
            total_label_acc = total_label_acc * 1.0 / total_seen
            total_seg_acc = total_seg_acc * 1.0 / total_seen

            test_loss_sum, test_label_acc_sum, test_label_loss_sum, test_seg_loss_sum, test_seg_acc_sum = sess.run(\
                    [total_test_loss_sum_op, label_test_acc_sum_op, label_test_loss_sum_op, seg_test_loss_sum_op, seg_test_acc_sum_op], \
                    feed_dict={total_testing_loss_ph: total_loss, label_testing_loss_ph: total_label_loss, \
                    seg_testing_loss_ph: total_seg_loss, label_testing_acc_ph: total_label_acc, seg_testing_acc_ph: total_seg_acc})

            test_writer.add_summary(test_loss_sum, (epoch_num+1) * num_train_file-1)
            test_writer.add_summary(test_label_loss_sum, (epoch_num+1) * num_train_file-1)
            test_writer.add_summary(test_seg_loss_sum, (epoch_num+1) * num_train_file-1)
            test_writer.add_summary(test_label_acc_sum, (epoch_num+1) * num_train_file-1)
            test_writer.add_summary(test_seg_acc_sum, (epoch_num+1) * num_train_file-1)

            printout(flog, '\tTesting Total Mean_loss: %f' % total_loss)
            printout(flog, '\t\tTesting Label Mean_loss: %f' % total_label_loss)
            printout(flog, '\t\tTesting Label Accuracy: %f' % total_label_acc)
            printout(flog, '\t\tTesting Seg Mean_loss: %f' % total_seg_loss)
            printout(flog, '\t\tTesting Seg Accuracy: %f' % total_seg_acc)

            for cat_idx in range(NUM_CATEGORIES):
                if total_seen_per_cat[cat_idx] > 0:
                    printout(flog, '\n\t\tCategory %s Object Number: %d' % (all_obj_cats[cat_idx][0], total_seen_per_cat[cat_idx]))
                    printout(flog, '\t\tCategory %s Label Accuracy: %f' % (all_obj_cats[cat_idx][0], total_label_acc_per_cat[cat_idx]/total_seen_per_cat[cat_idx]))
                    printout(flog, '\t\tCategory %s Seg Accuracy: %f' % (all_obj_cats[cat_idx][0], total_seg_acc_per_cat[cat_idx]/total_seen_per_cat[cat_idx]))

            return (total_loss, total_label_loss, total_label_acc, total_seg_loss, total_seg_acc)

        def visualize_one_epoch(epoch, table, vis_no = 3):
            if param_config.get_value('system', 'dataset') == 'shapenet': #FLAGS.dataset == 'shapenet':
                is_training = False

                total_loss = 0.0
                total_label_loss = 0.0
                total_seg_loss = 0.0
                total_label_acc = 0.0
                total_seg_acc = 0.0
                total_seen = 0

                total_label_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
                total_seg_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
                total_seen_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.int32)

                predictions = []
                ground_truths = []

                vis_indices = [1,2,21]

                cur_data, cur_labels, cur_seg, cur_pos = provider.loadDataFile_with_seg(batch_size, is_training=False, start_idx = 0, visualize=True)
                
                begidx = 0
                endidx = batch_size

                np.squeeze(cur_labels)

                cur_labels_one_hot = convert_label_to_one_hot(cur_labels)

                feed_dict = {
                    pointclouds_ph: cur_data[begidx: endidx, ...], 
                    labels_ph: cur_labels[begidx: endidx, ...], 
                    input_label_ph: cur_labels_one_hot[begidx: endidx, ...], 
                    seg_ph: cur_seg[begidx: endidx, ...],
                    is_training_ph: is_training, 
                    }

                label_pred_val, seg_pred_val, pred_seg_res \
                    = sess.run([labels_pred, seg_pred, per_instance_seg_pred_res], \
                    feed_dict=feed_dict)

                for idx in vis_indices:
                    ground_truths.append(
                            wandb.Object3D(np.hstack([cur_pos[idx], cur_seg[idx].reshape(-1, 1)]))
                        )

                    predictions.append(
                            wandb.Object3D(np.hstack([cur_pos[idx], pred_seg_res[idx].reshape(-1,1)]))
                        )

                # for i in range(vis_no):

                #     begidx = 0
                #     endidx = batch_size

                    
                #     # only load 1 graph at once
                #     cur_data, cur_labels, cur_seg, cur_pos = provider.loadDataFile_with_seg(32, is_training=False, start_idx = i * 32, visualize=True)

                #     ground_truths.append(
                #         wandb.Object3D(np.hstack([cur_pos[0], cur_seg[0].reshape(-1, 1)]))
                #     )

                #     #cur_labels = cur_labels.reshape([1,-1])
                #     np.squeeze(cur_labels)

                #     cur_labels_one_hot = convert_label_to_one_hot(cur_labels)

                #     # print(f'cur_data.shape {cur_data.shape}')
                #     # print(f'cur_labels.shape {cur_labels.shape}')
                #     # print(f'cur_labels_one_hot.shape {cur_labels_one_hot.shape}')
                #     # print(f'seg_ph.shape {seg_ph.shape}')

                #     feed_dict = {
                #         pointclouds_ph: cur_data[begidx: endidx, ...], 
                #         labels_ph: cur_labels[begidx: endidx, ...], 
                #         input_label_ph: cur_labels_one_hot[begidx: endidx, ...], 
                #         seg_ph: cur_seg[begidx: endidx, ...],
                #         is_training_ph: is_training, 
                #         }
                    
                #     # labels = ['labels_ph', 'input_label_ph', 'seg_ph']
                #     # for idx, val in enumerate(feed_dict.values()):
                #     #     if idx > 0 and idx < 4:
                #     #         print(labels[idx - 1], val)

                #     # feed_dict = {
                #     #         pointclouds_ph: cur_data, 
                #     #         labels_ph: cur_labels, 
                #     #         input_label_ph: cur_labels_one_hot, 
                #     #         seg_ph: cur_seg,
                #     #         is_training_ph: is_training, 
                #     #         }

                #     # loss_val, label_loss_val, seg_loss_val, per_instance_label_loss_val, \
                #     #         per_instance_seg_loss_val, label_pred_val, seg_pred_val, pred_seg_res \
                #     #         = sess.run([loss, label_loss, seg_loss, per_instance_label_loss, \
                #     #         per_instance_seg_loss, labels_pred, seg_pred, per_instance_seg_pred_res], \
                #     #         feed_dict=feed_dict)
                    
                #     label_pred_val, seg_pred_val, pred_seg_res \
                #             = sess.run([labels_pred, seg_pred, per_instance_seg_pred_res], \
                #             feed_dict=feed_dict)

                #     # print(f'label_pred_val {label_pred_val}')
                #     # print(f'label_pred_val.shape {label_pred_val.shape}')
                #     # print(f'seg_pred_val {seg_pred_val}')
                #     # print(f'seg_pred_val.shape {seg_pred_val.shape}')
                #     # print(f'pred_seg_res {pred_seg_res}')
                #     # print(f'pred_seg_res.shape {pred_seg_res.shape}')
                    
                #     # print(f'np.hstack() {np.hstack([cur_pos[0], pred_seg_res[0].reshape(-1,1)])}')

                #     predictions.append(
                #         wandb.Object3D(np.hstack([cur_pos[0], pred_seg_res[0].reshape(-1,1)]))
                #     )

                    

                # Store 3D models every 5 epochs
                if ((epoch + 1) % 5 == 0):
                    table.add_data(
                        epoch, ground_truths, predictions
                    )
                return table
            else:
                return None

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n<<< Testing on the test dataset ...')
            total_loss, total_label_loss, total_label_acc, total_seg_loss, total_seg_acc = eval_one_epoch(epoch)

            if param_config.get_value('system', 'use_wandb') == 'True':
                table = visualize_one_epoch(epoch, table)

            if param_config.get_value('system', 'use_wandb') == 'True':
                wandb.log({
                "Eval/Total_Loss": total_loss,
                "Eval/Total_Label_Loss": total_label_loss,
                "Eval/Total_Label_Accuracy": total_label_acc,
                "Eval/Total_Seg_Loss": total_seg_loss,
                "Eval/Total_Seg_Accuracy": total_seg_acc
                })

            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))

            if param_config.get_value('system', 'dataset') == 'shapenet':
                train_file_idx = num_train_file
            else:
                train_file_idx = np.arange(0, len(train_file_list))
                np.random.shuffle(train_file_idx)

            total_loss, total_label_loss, total_label_acc, total_seg_loss, total_seg_acc = train_one_epoch(train_file_idx, epoch)

            if param_config.get_value('system', 'use_wandb') == 'True':
                wandb.log({
                "Train/Total_Loss": total_loss,
                "Train/Total_Label_Loss": total_label_loss,
                "Train/Total_Label_Accuracy": total_label_acc,
                "Train/Total_Seg_Loss": total_seg_loss,
                "Train/Total_Seg_Accuracy": total_seg_acc
                })

            if (epoch+1) % 10 == 0:
                cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch+1)+'.ckpt'))
                printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

            flog.flush()

        flog.close()

        if param_config.get_value('system', 'use_wandb') == 'True':
            wandb.log({"PredClass_vs_TrueClass": table})
            wandb.finish()

if __name__=='__main__':
    train()
