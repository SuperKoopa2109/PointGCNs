import os
import sys
import numpy as np
import h5py

import torch_geometric.transforms as T
#from torch_geometric.datasets import ShapeNet #ModelNet
from part_seg.dataset.shapenet_loader import ShapeNet
from torch_geometric.loader import DataLoader

from param_config import param_config

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = param_config.get_value('paths', 'BASE_DIR')
print(f'*********************************')
print(f'BASE DIRECTORY: {BASE_DIR}')
print(f'*********************************')
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

if param_config.get_value('system', 'dataset') == 'modelnet40': 
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))
elif param_config.get_value('system', 'dataset') == 'shapenet':
            
        # BASE_DIR_DATA = param_config.get_value('paths', 'BASE_DIR')
        DATASET = param_config.get_value('system', 'dataset')

        category = "Airplane"

        # train_dataset = ShapeNet(
        #                     root=os.path.join(BASE_DIR, 'data', DATASET), 
        #                     categories=category, 
        #                     split='trainval')
        
        # test_dataset = ShapeNet(
        #                     root=os.path.join(BASE_DIR, 'data', DATASET), 
        #                     categories=category, 
        #                     split='test')
        
        # TMP CHANGE !!!! 

        no_points_sampled = 2048
        batch_size = param_config.get_value('config', 'batchsize')

        train_dataset = ShapeNet(
                            root=os.path.join(BASE_DIR, 'data', DATASET), 
                            categories=category, 
                            pre_transform=T.Compose([
                                    T.FixedPoints(no_points_sampled,replace = False, allow_duplicates = True)
                                        ]),
                            split='trainval')

        test_dataset = ShapeNet(
                            root=os.path.join(BASE_DIR, 'data', DATASET), 
                            categories=category,
                            pre_transform=T.Compose([
                                    T.FixedPoints(no_points_sampled,replace = False, allow_duplicates = True)
                                        ]), 
                            split='test')

        val_dataset = ShapeNet(
            root=os.path.join(BASE_DIR, 'data', DATASET), 
            categories=category, 
            pre_transform=T.Compose([
                                    T.FixedPoints(no_points_sampled,replace = False, allow_duplicates = True)
                                        ]),
            split = "val"
        )

        # END TMP CHANGE !!!!


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    if param_config.get_value('system', 'RunningInCOLAB') == 'True':
        return [os.path.join(param_config.get_value('paths', 'REPO_NAME'), line.rstrip()) for line in open(list_filename)]
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename, is_training=True, max_points=2048, start_idx=0, visualize=False, vis_samples=None):
    # h5_filename can also be used as a batch_size for shapenet dataset from pytorch geometric
    # Load data for 
    if param_config.get_value('system', 'dataset') == 'shapenet': 
        global train_dataset
        global test_dataset

        global train_loader
        global test_loader
        global val_loader
        global vis_loader
        category = "Airplane"
        
        data = np.zeros([h5_filename, max_points, 3])
        label = np.zeros([h5_filename], dtype=np.int32)
        seg = np.zeros([h5_filename, max_points], dtype=np.int32)
        pos = np.zeros([h5_filename, max_points, 3]) # 3 dims for x,y,z coordinates
        
        # TMP CHANGE !!!! #
        batch_size = param_config.get_value('config', 'batchsize')

        if is_training:
            
            train_loader = DataLoader(
                train_dataset[start_idx:min(start_idx + h5_filename, len(train_dataset) - 1 )],
                batch_size=1,
                shuffle=True
            )

            train_iter = iter(train_loader)

            for idx in range(h5_filename):
                batch = next(train_iter)
                data_batch, label_batch, seg_batch = batch['x'].numpy(), batch['category'].numpy(), batch['y'].numpy()
                
                data[idx] = data_batch
                label[idx] = label_batch
                seg[idx] = seg_batch
            
            return (data, label, seg)

        else:
            
            if visualize:

                vis_sample_size = 3

                vis_indices = [1,2,21] #np.random.choice(range(len(val_dataset)), size = vis_sample_size, replace = False)
                
                vis_loader = DataLoader(
                    [val_dataset[idx] for idx in vis_indices],
                    batch_size = 1,
                    shuffle = False
                )

                vis_iter = iter(vis_loader)

                for idx in range(h5_filename):
                    batch = next(vis_iter)
                    data_batch, label_batch, seg_batch = batch['x'].numpy(), batch['category'].numpy(), batch['y'].numpy()
                    pos_batch = batch['pos'].numpy()
                    
                    data[idx] = data_batch
                    label[idx] = label_batch
                    seg[idx] = seg_batch
                    pos[idx] = pos_batch 

                return (data, label, seg, pos)

            else:

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=True
                )

                val_iter = iter(val_loader)

                for idx in range(h5_filename):
                    batch = next(val_iter)
                    data_batch, label_batch, seg_batch = batch['x'].numpy(), batch['category'].numpy(), batch['y'].numpy()
                    
                    data[idx] = data_batch
                    label[idx] = label_batch
                    seg[idx] = seg_batch
                
                return (data, label, seg)

        # END TMP CHANGE !!!! #

        if is_training:
            for i in range(start_idx, min(start_idx + h5_filename, len(train_dataset) - 1 )):
                data_batch, label_batch, seg_batch = train_dataset[i]['x'].numpy(), train_dataset[i]['category'].numpy(), train_dataset[i]['y'].numpy()
                if visualize:
                    pos_batch = train_dataset[i]['pos'].numpy()
                if data_batch.shape[0] > max_points:
                    data_batch = data_batch[:max_points]
                    seg_batch = seg_batch[:max_points]
                    if visualize:
                        pos_batch = pos_batch[:max_points]
                elif data_batch.shape[0] < max_points:
                    while data_batch.shape[0] < max_points:
                        # if pointcloud does not contain enough points, add a random duplicate -> should not be too bad, as there are very few cases like this; also not optimal though TODO
                        rnd_idx = np.random.randint(h5_filename)
                        data_batch, label_batch, seg_batch = train_dataset[rnd_idx]['x'].numpy(), train_dataset[rnd_idx]['category'].numpy(), train_dataset[rnd_idx]['y'].numpy()
                        if visualize:
                            pos_batch = train_dataset[rnd_idx]['pos'].numpy()
                        if data_batch.shape[0] > max_points:
                            data_batch = data_batch[:max_points]
                            seg_batch = seg_batch[:max_points]
                            if visualize:
                                pos_batch = pos_batch[:max_points]
                data[i - start_idx] = data_batch
                label[i - start_idx] = label_batch
                seg[i - start_idx] = seg_batch
                
                if visualize:
                    # print(f'pos_batch {pos_batch}')
                    # print(f'pos.shape {pos.shape}')
                    # print(f'pos_batch.shape {pos_batch.shape}')
                    pos[i-start_idx] = pos_batch

                if visualize:
                    return (data, label, seg, pos)
        else:
            for i in range(start_idx, min(start_idx + h5_filename, len(test_dataset) - 1 )):
                data_batch, label_batch, seg_batch = test_dataset[i]['x'].numpy(), test_dataset[i]['category'].numpy(), test_dataset[i]['y'].numpy()
                if visualize:
                    pos_batch = test_dataset[i]['pos'].numpy()
                if data_batch.shape[0] > max_points:
                    data_batch = data_batch[:max_points]
                    seg_batch = seg_batch[:max_points]
                    if visualize:
                        pos_batch = pos_batch[:max_points]
                elif data_batch.shape[0] < max_points:
                    while data_batch.shape[0] < max_points:
                        # if pointcloud does not contain enough points, add a random duplicate -> should not be too bad, as there are very few cases like this; also not optimal though TODO
                        rnd_idx = np.random.randint(h5_filename)
                        data_batch, label_batch, seg_batch = test_dataset[rnd_idx]['x'].numpy(), test_dataset[rnd_idx]['category'].numpy(), test_dataset[rnd_idx]['y'].numpy()
                        if visualize:
                            pos_batch = test_dataset[rnd_idx]['pos'].numpy()
                        if data_batch.shape[0] > max_points:
                            data_batch = data_batch[:max_points]
                            seg_batch = seg_batch[:max_points]
                            if visualize:
                                pos_batch = pos_batch[:max_points]
                data[i - start_idx] = data_batch
                label[i - start_idx] = label_batch
                seg[i - start_idx] = seg_batch

                if visualize:
                    # print('----')
                    # print(f'pos_batch {pos_batch}')
                    # print(f'pos_batch.shape {pos_batch.shape}')
                    
                    pos[i - start_idx] = pos_batch
                    # print(f'pos {pos}')
                    # print('----')

                if visualize:
                    return (data, label, seg, pos)
            
            

        # TODO: How to handle which FILE TO GET!!! 
        # if h5_filename[0] == 'train':
        #     train_dataset = ShapeNet(
        #                         root=os.path.join(BASE_DIR_DATA, 'data', DATASET), 
        #                         categories=category, 
        #                         split='trainval')
        # elif h5_filename[0] == 'test':
        #     train_dataset = ShapeNet(
        #                         root=os.path.join(BASE_DIR_DATA, 'data', DATASET), 
        #                         categories=category, 
        #                         split='test')
        
        # in this dataset each iteration over train_dataset will give a batch, which is equal to one pointcloud (here: one airplane)
        # data = []
        # label = []
        # seg = []
        # for batch in train_dataset:
        #     data_batch, label_batch, seg_batch = (batch['x'], batch['category'], batch['y'])
        #     data.append(data_batch)
        #     label.append(label_batch)
        #     seg.append(seg_batch)

        # BATCH_SIZE = len(train_dataset)
        # NUM_WORKERS = 1

        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=BATCH_SIZE,
        #     shuffle=True,
        #     num_workers=NUM_WORKERS
        # )
    
    else:
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename, is_training=True, max_points=2048, start_idx=0, visualize=False):
    return load_h5_data_label_seg(filename, is_training=is_training, max_points=max_points, start_idx=start_idx, visualize=visualize)
