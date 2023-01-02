#Dataset setting
    #find global batch has element in current node cache
    #output the data need to be loaded for each node here
    #Change the data in the first global batch of shuffle list
    #0-87632 train imgs
import time
import h5py
import mpi4py as MPI
import math
import numpy as np
import torch

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