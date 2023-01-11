#Loading experiment for autophaseNN
from __future__ import print_function
from typing import Text, TextIO
import json
import numpy as np
import os
import torch
import random
import argparse
import time
import socket
import math
import itertools
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import multiprocessing as mp
from ctypes import *
import h5py
import pickle
import functools
import operator

#MPI setting

def get_local_rank(required=False):
    """Get local rank from environment."""
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    if 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    if required:
        raise RuntimeError('Could not get local rank')
    return 0


def get_local_size(required=False):
    """Get local size from environment."""
    if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_SIZE'])
    if 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    if 'SLURM_NTASKS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    if required:
        raise RuntimeError('Could not get local size')
    return 1


def get_world_rank(required=False):
    """Get rank in world from environment."""
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    if 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    if required:
        raise RuntimeError('Could not get world rank')
    return 0


def get_world_size(required=False):
    """Get world size from environment."""
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    if required:
        raise RuntimeError('Could not get world size')
    return 1


# Set global variables for rank, local_rank, world size
try:
    from mpi4py import MPI

    with_ddp=True
    local_rank=get_local_rank()
    rank=get_world_rank()
    size=get_world_size()

    # Pytorch will look for these:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    # It will want the master address too, which we'll broadcast:
    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(2345)

    if local_rank == 0:
        print("This is GPU 0 from node: %s" %(socket.gethostname()))

except Exception as e:
    with_ddp=False
    local_rank = 0
    size = 1
    rank = 0
    print("MPI initialization failed!")
    print(e)


class CosmoFlowTransform:
    """Standard transformations for a single CosmoFlow sample."""

    def __init__(self, apply_log):
        """Set up the transform.

        apply_log: If True, log-transform the data, otherwise use
        mean normalization.

        """
        self.apply_log = apply_log

    def __call__(self, x):
        x = x.float()
        if self.apply_log:
            x.log1p_()
        else:
            x /= x.mean() / functools.reduce(operator.__mul__, x.size())
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

class CosDataset(torch.utils.data.Dataset):
    """Cosmoflow data."""

    SUBDIR_FORMAT = '{:03d}'

    def __init__(self, indices, data_dir,dataset_size,
                 transform=None, transform_y=None):
        """Set up the CosmoFlow HDF5 dataset.

        This expects pre-split universes per split_hdf5_cosmoflow.py.

        You may need to transpose the universes to make the channel
        dimension be first. It is up to you to do this in the
        transforms or preprocessing.

        The sample will be provided to transforms in int16 format.
        The target will be provided to transforms in float format.

        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.transform_y = transform_y
        base_universe_size=512
        if h5py is None:
            raise ImportError('HDF5 dataset requires h5py')
        # Load info from cached index.
        idx_filename = os.path.join(data_dir, 'idx')
        with open(idx_filename, 'rb') as f:
            idx_data = pickle.load(f)
        self.sample_base_filenames = idx_data['filenames']
        self.num_subdirs = idx_data['num_subdirs']
        self.num_splits = (base_universe_size // idx_data['split_size'])**3

        self.num_samples = len(self.sample_base_filenames) * self.num_splits
        if dataset_size is not None:
            self.num_samples = min(dataset_size, self.num_samples)
        #self.cached_data_idx = dict()
        #self.cache_size = cache_size
        self.rank = rank
        self.load_numbers = 0
        self.cache_load = 0
        self.indices = indices
        self.epoch = 0
        self.load_time = 0
        self.cache_time = 0
        self.num_splits = 64
        print("init called")
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_samples

    def set_epoch(self,epoch):
        self.epoch = epoch
        self.load_numbers = 0
        self.cache_load = 0
        self.load_time = 0
        self.cache_time = 0

    def set_step(self):
        self.load_numbers = 0
        self.cache_load = 0
        self.load_time = 0
        self.cache_time = 0
        #print(len(self.cached_data_idx))

    def getLoadNumber(self):
        return self.load_numbers

    def getCacheLoad(self):
        return self.cache_load

    
    def get_time(self):
        #print("returning time: %s, %s" %(self.load_time,self.cache_time))
        return self.load_time,self.cache_time

    def __getitem__(self, index):
        idx = int(self.indices[self.epoch][index])
        self.load_numbers += 1
        # Handle flat vs nested directory structure.
        base_index = idx // self.num_splits  # Base filename.
        split_index = idx % self.num_splits  # Split within the universe.
        load_time_start=time.perf_counter()
        #print("num_subdirs:"+str(self.num_subdirs))
        #print("base file num:"+str(len(self.sample_base_filenames)))
        if self.num_subdirs:
            subdir = CosDataset.SUBDIR_FORMAT.format(
                base_index // self.num_subdirs)
            filename = os.path.join(
                self.data_dir,
                subdir,
                self.sample_base_filenames[base_index]
                + f'_{split_index:03d}.hdf5')
            x_idx = 'split'
        else:
            filename = os.path.join(
                self.data_dir,
                self.sample_base_filenames[base_index]
                + f'_{split_index:03d}.hdf5')
            x_idx = 'full'
        with h5py.File(filename, 'r') as f:
            x, y = f[x_idx][:], f['unitPar'][:]
        # Convert to Tensors.
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.transform is not None:
            x = self.transform(x)
        if self.transform_y is not None:
            y = self.transform_y(y)
        self.load_time +=time.perf_counter()-load_time_start
        return x, y

######################Parameter setup###############################
data_path = '/path/to/data/cosmoUniverse_21688988_v1/train/'
device='cpu'
batch_size = 16
run_time = 1
nepochs=30
# load data
#dataname_list = '/home/sunbaixi/DemoTest/PtychoNN-master/TF2/au_cosmoflow/results/file_lists'
filelist = []
#total_train_size=744*64
total_train_size=47616
apply_log = True
#with open(dataname_list, "rb") as fp:
        #filelist = pickle.load(fp)
#print('number of available file:%d' % len(filelist))
# give training data size and filelist
#train_filelist = filelist[:total_train_size]
print('number of training:%d' % total_train_size)
DATA_PATH=data_path
BATCH_SIZE=batch_size
nsamples = total_train_size
GLOBAL_BATCH_SIZE=BATCH_SIZE*size
step_size = round(nsamples/GLOBAL_BATCH_SIZE)
if rank == 0:
    print("Will have %s steps." %step_size)
######################END Parameter setup###############################

#shuffle list
shuffle_list=np.zeros([nepochs,nsamples])
if run_time == 1:
    for epoch in range(nepochs):
        idx_arr = np.arange(nsamples)
        np.random.shuffle(idx_arr)
        #idx_arr = MPI.COMM_WORLD.bcast(idx_arr, root=0)
        #print(idx_arr)
        shuffle_list[epoch] = idx_arr
        #print(idx_arr)
        #arr_sharded = shard(ngpus, idx_arr)
        #arr_sharded is in the shape of [#gpus, #samples per gpu]
        #epoch_list_sharded[epoch]=arr_sharded
    #shuffle_list_tensor = torch.Tensor(shuffle_list)
    shuffle_list = MPI.COMM_WORLD.bcast(shuffle_list, root=0)
else:
    print('Shuffle list loading not yet supported')
transform = CosmoFlowTransform(apply_log)
train_data2=CosDataset(indices=shuffle_list, data_dir=DATA_PATH, dataset_size=total_train_size,transform=transform)
kwargs = {'num_workers': 4, 'pin_memory': True} if device == 'gpu' else {}
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data2, num_replicas=size, shuffle=False, rank=rank)
train_loader = torch.utils.data.DataLoader(
    train_data2, batch_size=BATCH_SIZE, sampler=train_sampler,  **kwargs)

times=[]
avg_time_each_step=[]
loads=[]
caches=[]
total_io_epochs=[]
load_start_time = time.perf_counter()
for epoch in range (nepochs):
    total_io=0
    print("Non schedule epoch: %s" %epoch)
    epoch_start_time=time.perf_counter()
    train_sampler.set_epoch(epoch)
    train_data2.set_epoch(epoch)
    #data_times = AverageTracker()
    for i, (x,y) in tqdm(enumerate(train_loader)):
        start_time = time.perf_counter()
        '''
        if args.device == "gpu":
            ft_images = ft_images.cuda() #Move everything to device
            amps = amps.cuda()
            phs = phs.cuda()
        '''
        #if 1==epoch:
            #print(ft_images.size())
            #print(ft_images)
        if epoch == 5:
            load_numbers = train_data2.getLoadNumber()
            cache_numbers = train_data2.getCacheLoad()
            loads.append(load_numbers)
            caches.append(cache_numbers)
        hdf5_time,cache_time = train_data2.get_time()
        #io_time = hdf5_time+cache_time
        #io_time_t = np.zeros(size)
        #io_time_ar = np.zeros_like(io_time_t)
        #io_time_t[rank] = io_time
        #MPI.COMM_WORLD.Allreduce(io_time_t, io_time_ar, op=MPI.MAX)
        #total_io += io_time_ar[0]
        total_io += hdf5_time+cache_time
        train_data2.set_step()
        load_time=time.perf_counter() - start_time
        #data_times.update(load_time)
    
    total_io_epochs.append(total_io)
    #dataset.clean_cache()
    epoch_time = time.perf_counter()-epoch_start_time
    if rank == 0:
        print("~Rank: %s, Time for each epoch without assignment: %s" %(rank, epoch_time))
        print("~Rank: %s, total io time for each epoch: %s" %(rank, total_io))
    #avg_time = (epoch_time/i)
    times.append(epoch_time)
    #avg_time_each_step.append(avg_time)
    #data_times.save(os.path.join("/home/bsun/DemoTest/PtychoNN-master/logs/", f'stats_data.txt'))
load_end_time = time.perf_counter()
total_loading_time=load_end_time-load_start_time

if rank==0:
    print("*******************************************")
    print("DDP? "+str(with_ddp))
    print("Number of Nodes used: "+str(size))
    print("Epoch total number: "+str(nepochs))
    print("Batch Size: "+str(BATCH_SIZE))
    #print("DataLoading time per epoch without assignment: {:.4f} seconds".format(total_loading_time))
    #print("DataLoading average time per step without assignment: "+str(avg_time_each_step))
    #print("Time for each epoch without assignment: "+str(times))
    #print("DataLoading time per epoch with assignment: {:.4f} seconds".format(total_loading_time_s))
    #print("DataLoading average time per step with assignment: "+str(avg_time_each_step_s))
    #print("Time for each epoch with assignment: "+str(times_s))
    print("*******************************************")

print("Rank: %s, DataLoading time per epoch without assignment: %s" %(rank, total_loading_time))
#print("Rank: %s, DataLoading average time per step without assignment: %s" %(rank, avg_time_each_step))
print("Rank: %s, Time for each epoch without assignment: %s" %(rank, times))
print("Rank: %s, total io time for each epoch: %s" %(rank, total_io_epochs))
print("Rank: %s, load number: %s" %(rank, loads))
print("Rank: %s, cache number: %s" %(rank, caches))


