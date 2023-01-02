from __future__ import print_function
from email.policy import default
import os
import argparse
import time
import socket
import math
import itertools
import random
import numpy as np
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
import re
import collections
import torch

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

#Parse input parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--cache_size', type=int, default=1280, metavar='N',
                    help='Cache size for each node')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 60)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                    help='Whether this is running on cpu or gpu')
parser.add_argument('--num_threads', default=0, help='set number of threads per worker', type=int)
parser.add_argument('--ngpus', default=1, help='set number of gpus', type=int)
parser.add_argument('--nnodes', default=1, help='set number of nodes', type=int)
parser.add_argument('--gpu_pernode', default=1, help='set number of gpu per node', type=int)
parser.add_argument('--run_time', default=2, help='load from PFS on different folder to avoid PFS caching', type=int)
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to saved shuffled lists')
parser.add_argument('--dtest_path', default='', type=str, metavar='PATH',
                    help='path to test data')
parser.add_argument('--dtrain_path', default='', type=str, metavar='PATH',
                    help='path to training data')
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
                    help='path to save model, profiler logs')
args = parser.parse_args()

#Parameter Settings
NGPUS = args.ngpus
LR = NGPUS * 1e-3
H,W = 128,128

#DDP backend setting
# What backend?  nccl on GPU, gloo on CPU
if args.device == "gpu": backend = 'nccl'
elif args.device == "cpu": backend = 'gloo'

if with_ddp:
    torch.distributed.init_process_group(
        backend=backend, init_method='env://')


torch.manual_seed(args.seed)

if args.device == 'gpu':
    # DDP: pin GPU to local rank.
    torch.cuda.set_device(int(local_rank))
    torch.cuda.manual_seed(args.seed)

if (args.num_threads!=0):
    torch.set_num_threads(args.num_threads)

if rank==0:
    print("Torch Thread setup: ")
    print(" Number of threads: ", torch.get_num_threads())

path = os.getcwd()

#Dataset
class ptychoDataset(torch.utils.data.Dataset):
    """PtychoNN dataset from APS"""
    def __init__(self,file_path,nsamples,rank,size,cache_size,indices,to_load,local_batch_size):
        self.file_path = file_path
        self.nsamples = nsamples
        self.bench_load_step=set()
        self.cached_data_idx = dict()
        self.prefetch_buffer = dict()
        self.cache_size = cache_size
        self.loc_batch_size=local_batch_size
        self.rank = rank
        self.size = size
        self.step = 0
        self.idx_to_load=to_load
        self.idx_to_load_total=to_load
        self.idx_extra_load=set()
        self.loaded=dict()
        self.load_numbers = 0
        self.cache_load = 0
        self.indices = indices
        self.to_load_per_call = 0
        self.no_load_after_call = 0
        self.epoch = 0
        self.not_using=set()
        self.idx_extra_load_total = set()
        self.num_call = 0
        self.load_time = 0
        self.cache_time = 0
        #self.nsteps = nsteps
        self.loaded_curr_step = set()
        self.dset = h5py.File(file_path, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        print("init called")
        
    def __len__(self):
        return self.nsamples

    def get_idx_to_load(self):
        return self.bench_load_step
    
    def set_epoch(self,epoch):
        self.epoch = epoch
        self.num_call = 0
        self.load_time = 0
        self.load_time = 0
        self.cache_time = 0
    
    def set_step(self,step):
        self.step=step
        self.load_time = 0
        self.cache_time = 0
        self.load_numbers = 0
        self.cache_load = 0
        self.to_load_per_call = 0
        self.num_call = 0
        if self.epoch > 0 and self.step < len(self.idx_to_load[self.epoch-1]):
            self.not_using = self.idx_to_load[self.epoch-1][self.step]
        if self.epoch > 0:
            if len(self.idx_to_load[self.epoch-1]) > self.step:
                self.idx_extra_load = set(list(self.idx_to_load[self.epoch-1][self.step])[self.rank::self.size])
                self.idx_extra_load_total = set(list(self.idx_to_load[self.epoch-1][self.step]))
                self.to_load_per_call = math.ceil(len(self.idx_extra_load_total)/self.loc_batch_size)
                self.no_load_after_call = len(self.idx_extra_load_total)
            else: 
                self.idx_extra_load = set()
        self.loaded_curr_step = set()
        self.bench_load_step = set()
    
    def getLoadNumber(self):
        return self.load_numbers
    
    def getCacheLoad(self):
        return self.cache_load

    def getItemBalancing(self,idx,flag):
        self.loaded_curr_step.add(idx)
        cached=False
        prefetched=False
        if idx in self.cached_data_idx.keys():
            cached = True
        if idx in self.prefetch_buffer.keys():
            prefetched = True
        if not cached and not prefetched:
            #if not flag:
            self.load_numbers += 1
            load_time_start=time.perf_counter()
            X_train=self.dset["X_train"][idx]
            Y_I_train=self.dset["Y_I_train"][idx]
            Y_phi_train=self.dset["Y_phi_train"][idx]
            X_train = X_train[np.newaxis,:,:]
            Y_I_train = Y_I_train[np.newaxis,:,:]
            Y_phi_train = Y_phi_train[np.newaxis,:,:]
            x=X_train[:,:,:,0]
            y1=Y_I_train[:,:,:,0]
            y2=Y_phi_train[:,:,:,0]
            self.load_time +=time.perf_counter()-load_time_start
            if flag:
                self.cached_data_idx[idx]=[x,y1,y2]
                if len(self.cached_data_idx) > self.cache_size:
                    if 0 == self.epoch:
                        idx_to_replace=list(self.cached_data_idx.keys())[0]
                    else:
                        idx_to_replace=list(self.cached_data_idx.keys())[0]
                        if self.step < len(self.idx_to_load[self.epoch-1]):
                            for k in self.cached_data_idx.keys():
                                if k in self.not_using and k!=idx:
                                    idx_to_replace = k
                                    break
                    self.cached_data_idx.pop(idx_to_replace)
            else:
                self.prefetch_buffer[idx]=[x,y1,y2]
            
        elif cached and not prefetched:
            self.cache_load += 1
            cache_time_start=time.perf_counter()
            x=self.cached_data_idx[idx][0]
            y1=self.cached_data_idx[idx][1]
            y2=self.cached_data_idx[idx][2]
            self.cache_time +=time.perf_counter()-cache_time_start
        elif prefetched and not cached:
            self.cache_load += 1
            cache_time_start=time.perf_counter()
            x=self.prefetch_buffer[idx][0]
            y1=self.prefetch_buffer[idx][1]
            y2=self.prefetch_buffer[idx][2]
            self.cache_time +=time.perf_counter()-cache_time_start
        return x,y1,y2

    def get_time(self):
        return self.load_time,self.cache_time

    def __getitem__(self,item):
        
        self.num_call += 1
        idx = int(self.indices[self.epoch][item])
        x_list=[]
        y1_list=[]
        y2_list=[]
        #normal load
        if self.epoch == 0:
            return self.getItemBalancing(idx,True)
        if self.to_load_per_call > 0 and self.num_call <= self.no_load_after_call and self.epoch > 0:
            for tt in range(self.to_load_per_call):
                if len(self.idx_extra_load) > 0:
                    tt_idx = int(self.idx_extra_load.pop())
                    x_temp,y1_temp,y2_temp = self.getItemBalancing(tt_idx,False)
                    x_list.append(x_temp)
                    y1_list.append(y1_temp)
                    y2_list.append(y2_temp)
        x,y1,y2 = self.getItemBalancing(idx,True)
        x_list.append(x)
        y1_list.append(y1)
        y2_list.append(y2)
        
        return x_list,y1_list,y2_list

    def clean_cache(self):
        self.cached_data_idx = dict()



#MODEL_SAVE_PATH = path +'/trained_model/'
MODEL_SAVE_PATH =args.root_path
#if (not os.path.isdir(MODEL_SAVE_PATH)):
    #os.mkdir(MODEL_SAVE_PATH)

ngpu_pernode=args.gpu_pernode
nnodes=args.nnodes
nepochs=args.epochs
LOCAL_BATCH_SIZE = args.batch_size
GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * size
cache_size = args.cache_size
nsamples=87632
ngpus=ngpu_pernode*nnodes

#collate function
def swift_collate(batch):
    numel = 0
    for xx in batch:
        if xx[0] is not None:
            numel += len(xx[0])
    tensor_x = torch.zeros(size=(numel,1,128,128))
    tensor_y1 = torch.zeros(size=(numel,1,128,128))
    tensor_y2 = torch.zeros(size=(numel,1,128,128))
    pointer=0
    for i, ele in enumerate(batch):
        if len(ele[0])==1: #First epoch
            tensor_x[pointer] += ele[0][0]
            tensor_y1[pointer] += ele[1][0]
            tensor_y2[pointer] += ele[2][0]
            pointer += 1
        else:
            for k in range(len(ele[0])):
                tensor_x[pointer] += ele[0][k]
                tensor_y1[pointer] += ele[1][k]
                tensor_y2[pointer] += ele[2][k]
                pointer += 1
    return tensor_x,tensor_y1,tensor_y2

X_test = np.load(os.path.join(args.dtest_path,"X_test_t2.npy"))
Y_I_test = np.load(os.path.join(args.dtest_path,"Y_I_test_t2.npy"))
Y_phi_test = np.load(os.path.join(args.dtest_path,"Y_phi_test_t2.npy"))
X_test = X_test[:,:,:,0]
Y_I_test = Y_I_test[:,:,:,0]
Y_phi_test = Y_phi_test[:,:,:,0]

X_test = X_test[:,np.newaxis,:,:]
Y_I_test = Y_I_test[:,np.newaxis,:,:]
Y_phi_test = Y_phi_test[:,np.newaxis,:,:]
print(X_test.shape)

X_test_tensor = torch.Tensor(X_test)
Y_I_test_tensor = torch.Tensor(Y_I_test)
Y_phi_test_tensor = torch.Tensor(Y_phi_test)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor,Y_I_test_tensor, Y_phi_test_tensor)
step_size = round(nsamples/GLOBAL_BATCH_SIZE)
kwargs = {'num_workers':0, 'pin_memory': True} if args.device == 'gpu' else {}
#Distributed sampler and data loader
#original epoch order
#scheduled epoch order
shuffle_list_sorted=np.load(os.path.join(args.save_path,'pshuffle_list_sorted_debug_gpu8.npy'))
with open(os.path.join(args.save_path,'idx_to_load_total_debug'), 'rb') as fp:   # Unpickling
    idx_to_load_total = pickle.load(fp)
'''With scheduling'''
train_data3=ptychoDataset(file_path=os.path.join(args.dtrain_path,'train_t2.h5'),rank=rank,size=size,nsamples=nsamples,cache_size=cache_size/size,indices=shuffle_list_sorted,to_load=idx_to_load_total,local_batch_size=LOCAL_BATCH_SIZE)
train_sampler_sorted = torch.utils.data.distributed.DistributedSampler(
    train_data3, num_replicas=size, shuffle=False, rank=rank)
train_loader_sorted = torch.utils.data.DataLoader(
    train_data3, batch_size=LOCAL_BATCH_SIZE, shuffle=False,sampler=train_sampler_sorted,collate_fn=swift_collate, **kwargs)
valid_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=size, shuffle=True, rank=rank)
valid_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=LOCAL_BATCH_SIZE, sampler=valid_sampler,  **kwargs)

nconv = 32
class recon_model(nn.Module):

    def __init__(self):
        super(recon_model, self).__init__()

        self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
          nn.Conv2d(in_channels=1, out_channels=nconv, kernel_size=3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv, nconv, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.MaxPool2d((2,2)),

          nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),          
          nn.ReLU(),
          nn.MaxPool2d((2,2)),

          nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),          
          nn.ReLU(),
          nn.MaxPool2d((2,2)),

          nn.Conv2d(nconv*4, nconv*8, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*8, nconv*8, 3, stride=1, padding=(1,1)),          
          nn.ReLU(),
          nn.MaxPool2d((2,2)),
          )

        self.decoder1 = nn.Sequential(

          nn.Conv2d(nconv*8, nconv*8, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*8, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),
            
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
          nn.Sigmoid() #Amplitude model
          )

        self.decoder2 = nn.Sequential(

          nn.Conv2d(nconv*8, nconv*8, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*8, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),
            
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
          nn.Tanh() #Phase model
          )
    
    def forward(self,x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        #Restore -pi to pi range
        ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp,ph

model = recon_model()


if args.device == 'gpu':
    # Move model to GPU.
    model.cuda()

if with_ddp:
    # wrap the model in DDP:
    model = DDP(model)
    

# DDP: scale learning rate by the number of GPUs.

#Optimizer details
iterations_per_epoch = np.floor(nsamples/GLOBAL_BATCH_SIZE)+1 #Final batch will be less than batch size
step_size = 6*iterations_per_epoch #Paper recommends 2-10 number of iterations, step_size is half cycle
if local_rank==0:
    print("LR step size is:", step_size, "which is every %d epochs" %(step_size/iterations_per_epoch))

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR/10, max_lr=LR, step_size_up=step_size,
                                              cycle_momentum=False, mode='triangular2')
#Function to update saved model if validation loss is minimum
def update_saved_model(model, model_path):
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    for f in os.listdir(model_path):
        os.remove(os.path.join(model_path, f))
    if (NGPUS>1):    
        torch.save(model.module.state_dict(),model_path+'1best_model_proposed_gpu'+str(NGPUS)+'.pth') #Have to save the underlying model else will always need 4 GPUs
    else:
        torch.save(model,model_path+'1best_model_proposed_gpu'+str(NGPUS)+'.pth')

def metric_average(val, name):
    if (with_ddp):
        # Sum everything and divide by total size:
        dist.all_reduce(val,op=dist.ReduceOp.SUM)
        val /= size
    else:
        pass
    return val

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

#Profiling
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=4),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(args.root_path,'/profiler')),
        record_shapes=True,
        with_stack=True)
load_num=[]
#ll_times=[]
def train(train_loader,metrics,epoch):
    total_io=0
    tot_loss = torch.tensor(0.0)
    loss_amp = torch.tensor(0.0)
    loss_ph = torch.tensor(0.0)
    if args.device == 'gpu':
        tot_loss = tot_loss.cuda()
        loss_amp = loss_amp.cuda()
        loss_ph = loss_ph.cuda()
    
    #data_times = AverageTracker()
    
    
    #total_time_arr=np.zeros([size,args.epochs])
    comp_time = 0
    step_time = 0
    host_to_device = 0
    #total_io_epochs=[]
    start_time = time.perf_counter()
    for i, (ft_images,amps,phs) in tqdm(enumerate(train_loader)):
        host_to_device_start=time.perf_counter()
        if args.device == "gpu":
            ft_images = ft_images.cuda() #Move everything to device
            amps = amps.cuda()
            phs = phs.cuda()
        host_to_device+=time.perf_counter()-host_to_device_start
        step_start = time.perf_counter()
        hdf5_time,cache_time = train_data3.get_time()
        io_time = hdf5_time+cache_time
        #io_time_t = np.zeros(size)
        #io_time_ar = np.zeros_like(io_time_t)
        #io_time_t[rank] = io_time
        #MPI.COMM_WORLD.Allreduce(io_time_t, io_time_ar, op=MPI.MAX)
        #total_io += io_time_ar[0]
        total_io += io_time
        
        pred_amps, pred_phs = model(ft_images) #Forward pass
        load_numbers = train_data3.getLoadNumber()
        load_num.append(load_numbers)
        train_data3.set_step(i+1)
        #Compute losses
        loss_a = criterion(pred_amps,amps) #Monitor amplitude loss
        loss_p = criterion(pred_phs,phs) #Monitor phase loss but only within support (which may not be same as true amp)
        loss = loss_a + loss_p #Use equiweighted amps and phase

        #Zero current grads and do backprop
        optimizer.zero_grad() 
        loss.backward()
        step_time = time.perf_counter() - step_start
        average_gradients(model)
        optimizer.step()
        #profiler.step()
        #prof.step()
        tot_loss += loss.detach().item()
        loss_amp += loss_a.detach().item()
        loss_ph += loss_p.detach().item()
        #Update the LR according to the schedule -- CyclicLR updates each batch
        scheduler.step()
        comp_time += step_time
        metrics['lrs'].append(scheduler.get_last_lr())
        
    metrics['io_time'].append(total_io)
    end_time = time.perf_counter()
    metrics['compTime'].append(comp_time)
    #metrics['loadTime'].append(data_loading)
    tot_loss_avg = metric_average(tot_loss, 'loss')
    loss_amp_avg = metric_average(loss_amp, 'loss')
    loss_ph_avg = metric_average(loss_ph, 'loss')
    #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
    metrics['losses'].append([tot_loss_avg/i,loss_amp_avg/i,loss_ph_avg/i]) 

def validate(validloader,metrics):
    tot_val_loss = torch.tensor(0.0)
    val_loss_amp = torch.tensor(0.0)
    val_loss_ph = torch.tensor(0.0)
    if args.device == 'gpu':
        tot_val_loss = tot_val_loss.cuda()
        val_loss_amp = val_loss_amp.cuda()
        val_loss_ph = val_loss_ph.cuda()
    for j, (ft_images,amps,phs) in enumerate(validloader):
        if args.device == "gpu":
            ft_images = ft_images.cuda()
            amps = amps.cuda()
            phs = phs.cuda()
        pred_amps, pred_phs = model(ft_images) #Forward pass
    
        val_loss_a = criterion(pred_amps,amps) 
        val_loss_p = criterion(pred_phs,phs)
        val_loss = val_loss_a + val_loss_p
    
        tot_val_loss += val_loss.detach().item()
        val_loss_amp += val_loss_a.detach().item()
        val_loss_ph += val_loss_p.detach().item()
    
    tot_val_loss_avg = metric_average(tot_val_loss, 'loss')
    val_loss_amp_avg = metric_average(val_loss_amp, 'loss')
    val_loss_ph_avg = metric_average(val_loss_ph, 'loss')
    metrics['val_losses'].append([tot_val_loss_avg/j,val_loss_amp_avg/j,val_loss_ph_avg/j])
  
  #Update saved model if val loss is lower
    if(tot_val_loss_avg/j<metrics['best_val_loss']) and rank==0:
        print("Saving improved model after Val Loss improved from %.5f to %.5f" %(metrics['best_val_loss'],tot_val_loss_avg/j))
        metrics['best_val_loss'] = tot_val_loss_avg/j
        update_saved_model(model, MODEL_SAVE_PATH)


metrics = {'io_time':[], 'compTime':[],'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf}
#metrics = {'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf}
dur = []
loss=[]
loss_amp=[]
loss_ph=[]
val_loss=[]
val_loss_amp=[]
val_loss_phase=[]
training_start_time = time.time()
comp_time_arr=np.zeros([size,args.epochs])
load_time_arr=np.zeros([size,args.epochs])

for epoch in range (args.epochs):
    #train_sampler.set_epoch(epoch)
    #train_sampler_sorted.set_epoch(epoch)
    valid_sampler.set_epoch(epoch)
    train_data3.set_epoch(epoch)
    train_loader_sorted.sampler.set_epoch(epoch)
    train_data3.set_step(0)
    #Set model to train mode
    model.train()
    #Training loop
    t0 = time.time()
    #prof.start()
    train(train_loader_sorted,metrics,epoch)
    #prof.stop()
    dur.append(time.time() - t0)
    #Switch model to eval mode
    model.eval()
    #Validation loop
    validate(valid_loader,metrics)
    loss.append(metrics['losses'][-1][0].item())
    loss_amp.append(metrics['losses'][-1][1].item())
    loss_ph.append(metrics['losses'][-1][2].item())
    val_loss.append(metrics['val_losses'][-1][0].item())
    val_loss_amp.append(metrics['val_losses'][-1][1].item())
    val_loss_phase.append(metrics['val_losses'][-1][2].item())
    if rank==0:
        print('Epoch: %d | FT  | Train Loss: %.5f | Val Loss: %.5f' %(epoch, metrics['losses'][-1][0], metrics['val_losses'][-1][0]))
        print('Epoch: %d | Amp | Train Loss: %.4f | Val Loss: %.4f' %(epoch, metrics['losses'][-1][1], metrics['val_losses'][-1][1]))
        print('Epoch: %d | Ph  | Train Loss: %.3f | Val Loss: %.3f' %(epoch, metrics['losses'][-1][2], metrics['val_losses'][-1][2]))
        print('Epoch: %d | Ending LR: %.6f ' %(epoch, metrics['lrs'][-1][0]))
total_training_time=time.time() - training_start_time

if rank==0:
    print("*******************************************")
    print("DDP? "+str(with_ddp))
    print("Number of GPUs used: "+str(NGPUS))
    print("Epoch total number: "+str(args.epochs))
    print("Local Batch Size: "+str(LOCAL_BATCH_SIZE)+" Learning Rate: "+str(LR))
    print("Minimum Validation Loss: "+str(np.amin(np.asarray(val_loss))))
    print("Total training time: {:.2f} seconds".format(total_training_time))
    print("Loss: "+str(loss))
    print("Loss amp: "+str(loss_amp))
    print("Loss phase: "+str(loss_ph))
    print("Validation Loss: "+str(val_loss))
    print("Validation Loss amp: "+str(val_loss_amp))
    print("Validation Loss phase: "+str(val_loss_phase))
    print("Time for each epoch: "+str(dur))
    print("Loading time for each rank each epoch: ")
    #print("*******************************************")

print("Total IO time Rank %s: %s" %(rank,metrics['io_time']))
print("Computation time Rank %s: %s" %(rank,metrics['compTime']))