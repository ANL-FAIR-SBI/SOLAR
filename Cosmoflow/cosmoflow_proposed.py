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

    def __init__(self, indices,rank,size, data_dir, dataset_size,cache_size, to_load, local_batch_size,transform=None, transform_y=None):
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
        self.rank = rank
        self.size = size
        self.load_numbers = 0
        self.cache_load = 0
        self.indices = indices
        self.epoch = 0
        self.load_time = 0
        self.cache_time = 0
        self.bench_load_step=set()
        self.cached_data_idx = dict()
        self.prefetch_buffer = dict()
        self.cache_size = cache_size
        self.loc_batch_size=local_batch_size
        self.step = 0
        self.idx_to_load=to_load
        self.idx_to_load_total=to_load
        self.idx_extra_load=set()
        self.loaded=dict()
        self.to_load_per_call = 0
        self.no_load_after_call = 0
        self.not_using=set()
        self.idx_extra_load_total = set()
        self.num_call = 0
        self.loaded_curr_step = set()
        self.num_splits = 64
        print("init called")

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_samples

    def set_epoch(self,epoch):
        self.epoch = epoch
        self.num_call = 0
        self.load_numbers = 0
        self.cache_load = 0
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
                #if self.step==3:
                    #print("Rank: %s, to load %s" %(rank, len(self.idx_extra_load_total)))
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
            load_time_start=time.perf_counter()
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
            if flag:
                self.cached_data_idx[idx]=[x,y]
                if len(self.cached_data_idx) > self.cache_size:
                    if 0 == self.epoch:
                        idx_to_replace=list(self.cached_data_idx.keys())[0]
                    else:
                        idx_to_replace=list(self.cached_data_idx.keys())[0]
                        if self.step < len(self.idx_to_load[self.epoch-1]):
                            for k in self.cached_data_idx.keys():
                                if k in self.not_using and k!=idx:
                                    idx_to_replace = k
                                    #self.idx_to_load[self.epoch-1][self.step].remove(k)
                                    break
                    self.cached_data_idx.pop(idx_to_replace)
            else:
                self.prefetch_buffer[idx]=[x,y]
            
        elif cached and not prefetched:
            self.cache_load += 1
            cache_time_start=time.perf_counter()
            x=self.cached_data_idx[idx][0]
            y=self.cached_data_idx[idx][1]
            self.cache_time +=time.perf_counter()-cache_time_start
        elif prefetched and not cached:
            self.cache_load += 1
            cache_time_start=time.perf_counter()
            x=self.prefetch_buffer[idx][0]
            y=self.prefetch_buffer[idx][1]
            self.cache_time +=time.perf_counter()-cache_time_start
        return x,y

    def get_time(self):
        #print("returning time: %s, %s" %(self.load_time,self.cache_time))
        return self.load_time,self.cache_time

    def __getitem__(self, index):
        
        self.num_call += 1
        idx = int(self.indices[self.epoch][index])
        #x,y1,y2=None,None,None
        x_list=[]
        y_list=[]
        #normal load
        #conditions: no in to load
        if self.epoch == 0:
            return self.getItemBalancing(idx,True)
        #if (idx in self.cached_data_idx.keys() or idx in self.prefetch_buffer.keys()):
        #extra load
        # conditions: in to load
        # test 1: use two buffers to observe the number loaded
        #print("1: %s, 2: %s, 3: %s" %(self.to_load_per_call > 0,self.num_call <= self.no_load_after_call,self.epoch > 0))
        if self.to_load_per_call > 0 and self.num_call <= self.no_load_after_call and self.epoch > 0:
            #self.idx_extra_load = list(self.idx_to_load[self.epoch-1][self.step])[rank::size]
            for tt in range(self.to_load_per_call):
                if len(self.idx_extra_load) > 0:
                    #t_idx = None
                    #for i in range(len(self.idx_extra_load)):
                    tt_idx = int(self.idx_extra_load.pop())
                    x_temp,y_temp = self.getItemBalancing(tt_idx,False)
                    x_list.append(x_temp)
                    y_list.append(y_temp)
        #if idx not in self.loaded_curr_step:
        if idx not in self.idx_extra_load_total:  
            x,y = self.getItemBalancing(idx,True)
            x_list.append(x)
            y_list.append(y)
        return x_list,y_list

collate_times=[]
def swift_collate(batch):
    collate_time_start=time.perf_counter()
    numel = 0
    for xx in batch:
        if xx[0] is not None:
            numel += len(xx[0])
    tensor_x = torch.zeros(size=(numel,4,128,128,128))
    tensor_y = torch.zeros(size=(numel,4))
    pointer=0
    for i, ele in enumerate(batch):
        if len(ele[0])==1: #First epoch
            tensor_x[pointer] += ele[0][0]
            tensor_y[pointer] += ele[1][0]
            pointer += 1
        else:
            for k in range(len(ele[0])):
                tensor_x[pointer] += ele[0][k]
                tensor_y[pointer] += ele[1][k]
                pointer += 1
    collate_time=time.perf_counter()-collate_time_start
    collate_times.append(collate_time)
    return tensor_x,tensor_y

######################Parameter setup###############################
data_path = '/path/to/idx/cosmoUniverse_21688988_v1/train/'
device='cpu'
batch_size = 16
run_time = 2
nepochs=100
cache_size=47616
#cache_size = 38550 #16proc 40GB
#cache_size = 30840 #32proc 16G
total_train_size=47616
saving_path='/path/to/idx/list'
apply_log = True
# load data
#dataname_list = os.path.join(data_path, DataSummary)
#filelist = []
#with open(dataname_list, 'r') as f:
    #txtfile = f.readlines()
#for i in range(len(txtfile)):
    #tmp = str(txtfile[i]).split('/')[-1]
    #tmp = tmp.split('\n')[0]

    #filelist.append(tmp)
#f.close()
#print('number of available file:%d' % total_train_size)
# give training data size and filelist
#train_filelist = filelist[:total_train_size]
print('number of training:%d' % total_train_size)
#DATA_ID=train_filelist
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
shuffle_list_sorted=np.zeros([nepochs,nsamples])


#Load Scheduling
    #generate shuffle list
    #compute cost and map to graph adj matrix
    #find shortest path

#modified cost: rank imbalancement - extras need to be loaded!
def generate_weight_matrix_cache_fifo_new(arr,local_batch_size,cache_size,size):
    #Input should be the sharded array with size [#epochs, #sample ]
    #For example: Assume that GPU0 and GPU1 is on node 1, GPU2 and GPU3 on node 2
    #print("array shape:", arr.shape)
    matrix=np.zeros([arr.shape[0],arr.shape[0]])
    for idx in range(arr.shape[0]):
        source_cache = arr[idx,-cache_size:].reshape(cache_size)
        for r in range(arr.shape[0]):
            #Compute the distance between each epochs
            #Epoch idx is the source and epoch r is the target
            cost=0
            rank_count=np.zeros(size)
            if not idx==r:
                curr_samples = arr[r,:cache_size].reshape(cache_size)
                d = dict()
                for i in range(source_cache.shape[0]):
                    d[source_cache[i]] = 0
                for s in range(curr_samples.shape[0]):
                    curr_rank = s % size
                    if not curr_samples[s] in d.keys():
                        cost += 1
                    #else:
                        #if rank_count[curr_rank] > local_batch_size:
                            #cost += 1
                        #rank_count[curr_rank] += 1
                matrix[idx][r] = int(cost//size)
            
            else:
                matrix[idx][r] = np.nan
    
    return matrix


def shard(ngpus,array):
    list_result=np.zeros([ngpus,math.ceil(array.shape[0]/ngpus)])
    for g in range(ngpus):
        ii=0
        list_result[g]=array[g::ngpus]
    return list_result  

#https://github.com/marcoscastro/tsp_pso/blob/master/tsp_pso.py
# encoding:utf-8

'''
    Solution for Travelling Salesman Problem using PSO (Particle Swarm Optimization)
    Discrete PSO for TSP
    References: 
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.258.7026&rep=rep1&type=pdf
        http://www.cs.mun.ca/~tinayu/Teaching_files/cs4752/Lecture19_new.pdf
        http://www.swarmintelligence.org/tutorials.php
    References are in the folder "references" of the repository.
'''

from operator import attrgetter
import random, sys, time, copy


# class that represents a graph
class Graph:

    def __init__(self, amount_vertices):
        self.edges = {} # dictionary of edges
        self.vertices = set() # set of vertices
        self.amount_vertices = amount_vertices # amount of vertices


    # adds a edge linking "src" in "dest" with a "cost"
    def addEdge(self, src, dest, cost = 0):
    # checks if the edge already exists
        if not self.existsEdge(src, dest):
            self.edges[(src, dest)] = cost
            self.vertices.add(src)
            self.vertices.add(dest)


    # checks if exists a edge linking "src" in "dest"
    def existsEdge(self, src, dest):
        return (True if (src, dest) in self.edges else False)


    # shows all the links of the graph
    def showGraph(self):
        print('Showing the graph:\n')
        for edge in self.edges:
            print('%d linked in %d with cost %d' % (edge[0], edge[1], self.edges[edge]))

    # returns total cost of the path
    def getCostPath(self, path):

        total_cost = 0
        for i in range(self.amount_vertices - 1):
            total_cost += self.edges[(path[i], path[i+1])]

        # add cost of the last edge
        #total_cost += self.edges[(path[self.amount_vertices - 1], path[0])]
        return total_cost


    # gets random unique paths - returns a list of lists of paths
    def getRandomPaths(self, max_size):

        random_paths, list_vertices = [], list(self.vertices)

        initial_vertice = random.choice(list_vertices)
        if initial_vertice not in list_vertices:
            print('Error: initial vertice %d not exists!' % initial_vertice)
            sys.exit(1)

        list_vertices.remove(initial_vertice)
        list_vertices.insert(0, initial_vertice)

        for i in range(max_size):
            list_temp = list_vertices[1:]
            random.shuffle(list_temp)
            list_temp.insert(0, initial_vertice)

            if list_temp not in random_paths:
                random_paths.append(list_temp)

        return random_paths


# class that represents a complete graph
class CompleteGraph(Graph):

    # generates a complete graph
    def generates(self):
        for i in range(self.amount_vertices):
            for j in range(self.amount_vertices):
                if i != j:
                    weight = random.randint(1, 10)
                    self.addEdge(i, j, weight)


# class that represents a particle
class Particle:

    def __init__(self, solution, cost):

        # current solution
        self.solution = solution

        # best solution (fitness) it has achieved so far
        self.pbest = solution

        # set costs
        self.cost_current_solution = cost
        self.cost_pbest_solution = cost

        # velocity of a particle is a sequence of 4-tuple
        # (1, 2, 1, 'beta') means SO(1,2), prabability 1 and compares with "beta"
        self.velocity = []

    # set pbest
    def setPBest(self, new_pbest):
        self.pbest = new_pbest

    # returns the pbest
    def getPBest(self):
        return self.pbest

    # set the new velocity (sequence of swap operators)
    def setVelocity(self, new_velocity):
        self.velocity = new_velocity

    # returns the velocity (sequence of swap operators)
    def getVelocity(self):
        return self.velocity

    # set solution
    def setCurrentSolution(self, solution):
        self.solution = solution

    # gets solution
    def getCurrentSolution(self):
        return self.solution

    # set cost pbest solution
    def setCostPBest(self, cost):
        self.cost_pbest_solution = cost

    # gets cost pbest solution
    def getCostPBest(self):
        return self.cost_pbest_solution

    # set cost current solution
    def setCostCurrentSolution(self, cost):
        self.cost_current_solution = cost

    # gets cost current solution
    def getCostCurrentSolution(self):
        return self.cost_current_solution

    # removes all elements of the list velocity
    def clearVelocity(self):
        del self.velocity[:]


# PSO algorithm
class PSO:

    def __init__(self, graph, iterations, size_population, beta=1, alfa=1):
        self.graph = graph # the graph
        self.iterations = iterations # max of iterations
        self.size_population = size_population # size population
        self.particles = [] # list of particles
        self.beta = beta # the probability that all swap operators in swap sequence (gbest - x(t-1))
        self.alfa = alfa # the probability that all swap operators in swap sequence (pbest - x(t-1))

        # initialized with a group of random particles (solutions)
        solutions = self.graph.getRandomPaths(self.size_population)

        # checks if exists any solution
        if not solutions:
            print('Initial population empty! Try run the algorithm again...')
            sys.exit(1)

        # creates the particles and initialization of swap sequences in all the particles
        for solution in solutions:
            # creates a new particle
            particle = Particle(solution=solution, cost=graph.getCostPath(solution))
            # add the particle
            self.particles.append(particle)

        # updates "size_population"
        self.size_population = len(self.particles)


    # set gbest (best particle of the population)
    def setGBest(self, new_gbest):
        self.gbest = new_gbest

    # returns gbest (best particle of the population)
    def getGBest(self):
        return self.gbest


    # shows the info of the particles
    def showsParticles(self):

        print('Showing particles...\n')
        for particle in self.particles:
            print('pbest: %s\t|\tcost pbest: %d\t|\tcurrent solution: %s\t|\tcost current solution: %d' \
                % (str(particle.getPBest()), particle.getCostPBest(), str(particle.getCurrentSolution()),
                            particle.getCostCurrentSolution()))
        print('')


    def run(self):

        # for each time step (iteration)
        for t in range(self.iterations):

            # updates gbest (best particle of the population)
            self.gbest = min(self.particles, key=attrgetter('cost_pbest_solution'))

            # for each particle in the swarm
            for particle in self.particles:

                particle.clearVelocity() # cleans the speed of the particle
                temp_velocity = []
                solution_gbest = copy.copy(self.gbest.getPBest()) # gets solution of the gbest
                solution_pbest = particle.getPBest()[:] # copy of the pbest solution
                solution_particle = particle.getCurrentSolution()[:] # gets copy of the current solution of the particle

                # generates all swap operators to calculate (pbest - x(t-1))
                for i in range(self.graph.amount_vertices):
                    if solution_particle[i] != solution_pbest[i]:
                        # generates swap operator
                        swap_operator = (i, solution_pbest.index(solution_particle[i]), self.alfa)

                        # append swap operator in the list of velocity
                        temp_velocity.append(swap_operator)

                        # makes the swap
                        aux = solution_pbest[swap_operator[0]]
                        solution_pbest[swap_operator[0]] = solution_pbest[swap_operator[1]]
                        solution_pbest[swap_operator[1]] = aux

                # generates all swap operators to calculate (gbest - x(t-1))
                for i in range(self.graph.amount_vertices):
                    if solution_particle[i] != solution_gbest[i]:
                        # generates swap operator
                        swap_operator = (i, solution_gbest.index(solution_particle[i]), self.beta)

                        # append swap operator in the list of velocity
                        temp_velocity.append(swap_operator)

                        # makes the swap
                        aux = solution_gbest[swap_operator[0]]
                        solution_gbest[swap_operator[0]] = solution_gbest[swap_operator[1]]
                        solution_gbest[swap_operator[1]] = aux

                # updates velocity
                particle.setVelocity(temp_velocity)

                # generates new solution for particle
                for swap_operator in temp_velocity:
                    if random.random() <= swap_operator[2]:
                        # makes the swap
                        aux = solution_particle[swap_operator[0]]
                        solution_particle[swap_operator[0]] = solution_particle[swap_operator[1]]
                        solution_particle[swap_operator[1]] = aux

                # updates the current solution
                particle.setCurrentSolution(solution_particle)
                # gets cost of the current solution
                cost_current_solution = self.graph.getCostPath(solution_particle)
                # updates the cost of the current solution
                particle.setCostCurrentSolution(cost_current_solution)

                # checks if current solution is pbest solution
                if cost_current_solution < particle.getCostPBest():
                    particle.setPBest(solution_particle)
                    particle.setCostPBest(cost_current_solution)


#epoch_list_sharded=np.zeros([nepochs,ngpus,math.ceil(nsamples/ngpus)])
if run_time == 1:
    for epoch in range(nepochs):
        idx_arr = np.arange(nsamples)
        np.random.shuffle(idx_arr)
        #idx_arr = MPI.COMM_WORLD.bcast(idx_arr, root=0)
        shuffle_list[epoch] = idx_arr
        #arr_sharded = shard(ngpus, idx_arr)
        #arr_sharded is in the shape of [#gpus, #samples per gpu]
        #epoch_list_sharded[epoch]=arr_sharded
    
    np.save(saving_path+'shuffle_list_debug.npy', shuffle_list)
    idx_to_load_total=[]
    if 0 == rank:
        mat_start_time = time.perf_counter()
        matrix_result = generate_weight_matrix_cache_fifo_new(shuffle_list,BATCH_SIZE,cache_size,size)
        mat_time = time.perf_counter() - mat_start_time
        print("Cost matrix done! Time: ",mat_time)
        np.save(saving_path+'matrix_result.npy', matrix_result)
        #Print the original cost
        cost_default1=0
        for e in range(nepochs-1):
            cost_default1 +=  matrix_result[e][e+1]
        print("Original cost: %s" %(cost_default1))
        # creates the Graph instance
        graph = Graph(amount_vertices=nepochs)
        for i in range(nepochs):
            for j in range(nepochs):
                if not i == j:
                    weight = matrix_result[i][j]
                    graph.addEdge(i,j,weight)
                    
        # creates a PSO instance
        pso = PSO(graph, iterations=1000, size_population=nepochs, beta=1, alfa=0.9)
        pso_start_time = time.perf_counter()
        pso.run() # runs the PSO algorithm
        pso_time = time.perf_counter() - pso_start_time
        print("PSO done! Time: ",pso_time)
        res_path=pso.getGBest().getPBest()
        pso_cost = pso.getGBest().getCostPBest()
        #Get the scheduled shuffle list
        cost_def = 0
        for i in range(nepochs-1):
            print(matrix_result[i][i+1])
            cost_def += matrix_result[i][i+1]
        print(res_path)
        cost_pso=0
        for j in range(nepochs-1):
            u=int(res_path[j])
            v=int(res_path[j+1])
            cost_pso += matrix_result[u][v]
        print("Default cost: %s, PSO cost: %s" %(cost_def,pso_cost))
        
        for e_i in range(nepochs):
            curr_idx = res_path[e_i]
            shuffle_list_sorted[e_i] = shuffle_list[curr_idx] #[nepoch,sample]
            #epoch_list_sharded_sorted[e_i] = epoch_list_sharded[curr_idx] #[nepoch,ngpu,sample_p_gpu]
        #Schedule the node swaps
        np.save(saving_path+'shuffle_list_debug_pso.npy', shuffle_list_sorted)
        #local_cache_size = cache_size//size
        num_steps_in_cache = int(cache_size//GLOBAL_BATCH_SIZE)
        nswap_list=[]
        #find from each node of source in the whole target first cache
        #Use the samples in the cache to generate the local batch
        scheduling_start = time.perf_counter()
        for i in range(nepochs-1):
            #i and i+1 loading
            #source_sample = epoch_list_sharded_sorted[i,:,-local_cache_size:].reshape(size*local_cache_size)
            source_sample = shuffle_list_sorted[i,-cache_size:]

            idx_to_load_epoch=[]
            for step in range(num_steps_in_cache):
                #curr_gpus = np.arange((n_s*gpus_per_node),((n_s+1)*gpus_per_node))
                #target_sample = epoch_list_sharded_sorted[i+1,:,:local_batch_size].reshape(size*local_batch_size)
                st = step * GLOBAL_BATCH_SIZE
                ed = st + GLOBAL_BATCH_SIZE
                #source_sample = shuffle_list_sorted[i,st:ed]
                target_sample = shuffle_list_sorted[i+1,st:ed]
                idx_to_load=[]
                idx_to_load_step=set()
                pointer_right=GLOBAL_BATCH_SIZE-1
                next_avail=np.arange(size)
                temp_list=np.zeros(GLOBAL_BATCH_SIZE)
                temp_list[:] = np.nan
                sevisity=0
                #Fill in the samples load from cache first
                for s in range(GLOBAL_BATCH_SIZE):
                    idx_in_source = np.where(source_sample == target_sample[s])
                    temp_var=-1
                    if 1 == np.size(idx_in_source):
                        rank_in_source = idx_in_source[0] % size
                        idx_in_temp = next_avail[rank_in_source]
                        if idx_in_temp < GLOBAL_BATCH_SIZE:
                            temp_list[idx_in_temp] = target_sample[s]
                            next_avail[rank_in_source] += size
                        else:
                            avail_idx = np.argwhere(np.isnan(temp_list))
                            de_idx = avail_idx[0]
                            de_idx_rank = de_idx % size
                            temp_list[de_idx] = target_sample[s]
                            next_avail[de_idx_rank] += size
                            sevisity += 1
                            idx_to_load_step.add(target_sample[s])
                    else:
                        idx_to_load.append(target_sample[s])
                        idx_to_load_step.add(target_sample[s])
                
                #Now, fill in the samples to load
                idx_to_fit=np.argwhere(np.isnan(temp_list))
                idx_shaped = idx_to_fit.reshape(np.size(idx_to_fit))
                temp_list[idx_shaped] = idx_to_load
                #print("Left",pointer_left)
                #print("right",pointer_right)
                idx_to_load_epoch.append(idx_to_load_step)
                shuffle_list_sorted[i+1,st:ed] = temp_list #Replace original order with the new order     
            idx_to_load_total.append(idx_to_load_epoch)
        scheduling_time = time.perf_counter() - scheduling_start
        print("scheduling done!, time: %s", scheduling_time)
    #Boardcast the shuffle lists before using them
    shuffle_list_sorted = MPI.COMM_WORLD.bcast(shuffle_list_sorted, root=0)
    idx_to_load_total = MPI.COMM_WORLD.bcast(idx_to_load_total, root=0)
    np.save(saving_path+'shuffle_list_sorted_debug.npy', shuffle_list_sorted)
    with open(saving_path+"idx_to_load_total_debug", "wb") as fp:
        pickle.dump(idx_to_load_total,fp)
else:
    shuffle_list = np.load(saving_path+'shuffle_list_debug.npy')
    '''
    for epoch in range(nepochs):
        idx_arr = shuffle_list[epoch]
        #idx_arr = MPI.COMM_WORLD.bcast(idx_arr, root=0)
        arr_sharded = shard(ngpus, idx_arr)
        #arr_sharded is in the shape of [#gpus, #samples per gpu]
        #epoch_list_sharded[epoch]=arr_sharded
    '''
    shuffle_list_sorted = np.load(saving_path+'shuffle_list_sorted_debug.npy')
    #shuffle_list_sorted = np.load('/grand/sbi-fair/ptychoNN_data/large/shuffle_lists/shuffle_list_debugv4.npy')
    with open(saving_path+"idx_to_load_total_debug", "rb") as fp:   # Unpickling
        idx_to_load_total = pickle.load(fp)

transform = CosmoFlowTransform(apply_log)
train_data2=CosDataset(indices=shuffle_list_sorted,
                        rank=rank,
                        size=size,
                        data_dir=DATA_PATH,
                        dataset_size=total_train_size,
                        cache_size=cache_size/size,
                        to_load=idx_to_load_total,
                        local_batch_size=BATCH_SIZE,
                        transform=transform
                        )
kwargs = {'num_workers': 4, 'pin_memory': True} if device == 'gpu' else {}
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data2, num_replicas=size, shuffle=False, rank=rank)
train_loader = torch.utils.data.DataLoader(
    train_data2, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=swift_collate,  **kwargs)


times=[]
avg_time_each_step=[]
loads=[]
caches=[]
total_io_epochs=[]
cost_load=0
load_start_time = time.perf_counter()
for epoch in range (nepochs):
    total_io=0
    print("Schedule epoch: %s" %epoch)
    epoch_start_time=time.perf_counter()
    train_sampler.set_epoch(epoch)
    train_data2.set_epoch(epoch)
    train_data2.set_step(0)
    #data_times = AverageTracker()
    for i, (x,y) in tqdm(enumerate(train_loader)):
        start_time = time.perf_counter()
        '''
        if args.device == "gpu":
            ft_images = ft_images.cuda() #Move everything to device
            amps = amps.cuda()
            phs = phs.cuda()
        '''
        load_time=time.perf_counter() - start_time
        #if 1==epoch:
            #print(ft_images.size())
            #print(ft_images)
        if epoch == 5:
            load_numbers = train_data2.getLoadNumber()
            cache_numbers = train_data2.getCacheLoad()
            loads.append(load_numbers)
            caches.append(cache_numbers)
        hdf5_time,cache_time = train_data2.get_time()
        io_time = hdf5_time+cache_time
        io_time_t = np.zeros(size)
        io_time_ar = np.zeros_like(io_time_t)
        io_time_t[rank] = io_time
        MPI.COMM_WORLD.Allreduce(io_time_t, io_time_ar, op=MPI.MAX)
        total_io += io_time_ar[0]
        train_data2.set_step(i+1)
        #data_times.update(load_time)
    
    total_io_epochs.append(total_io)
    #dataset.clean_cache()
    epoch_time = time.perf_counter()-epoch_start_time
    if rank == 0:
        print("~Rank: %s, Time for each epoch without assignment: %s" %(rank, epoch_time))
        print("~Rank: %s, total io time for each epoch: %s" %(rank, total_io))
    avg_time = (epoch_time/(i+1))
    times.append(epoch_time)
    avg_time_each_step.append(avg_time)
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


