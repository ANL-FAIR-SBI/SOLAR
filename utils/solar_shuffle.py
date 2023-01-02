from PSO import *
import argparse
import time
import math
import pickle
import numpy as np
import os

#Parse input parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--size', type=int, default=16, metavar='N',
                    help='The number of processes using to train')
parser.add_argument('--gpu_pernode', default=1, help='set number of gpu per node', type=int)
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='The number of processes using to train')
parser.add_argument('--nnodes', default=1, help='set number of nodes', type=int)
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 60)')
parser.add_argument('--cache_size', default=1, help='set the size of cache', type=int)
parser.add_argument('--ntrain', default=1, help='number of trainging samples', type=int)
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save shuffled lists')        
parser.add_argument('--run_time', default=1, help='run time==1?compute:load and pass', type=int)
args = parser.parse_args()

size=args.size
ngpu_pernode=args.gpu_pernode
nnodes=args.nnodes
nepochs=args.epochs
LOCAL_BATCH_SIZE = args.batch_size
GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * size
cache_size = args.cache_size
nsamples=args.ntrain
ngpus=ngpu_pernode*nnodes
NGPUS=ngpus

shuffle_list=np.zeros([args.epochs,nsamples])
shuffle_list_sorted=np.zeros([args.epochs,nsamples])


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

if args.run_time == 1:
    for epoch in range(nepochs):
        idx_arr = np.arange(nsamples)
        np.random.shuffle(idx_arr)
        shuffle_list[epoch] = idx_arr
    
    np.save(os.path.join(args.save_path,'original_shuffle_list.npy'), shuffle_list)
    idx_to_load_total=[]
    mat_start_time = time.perf_counter()
    matrix_result = generate_weight_matrix_cache_fifo_new(shuffle_list,LOCAL_BATCH_SIZE,cache_size,size)
    mat_time = time.perf_counter() - mat_start_time
    print("Cost matrix done! Time: ",mat_time)
    np.save(os.path.join(args.save_path,'matrix_result.npy'), matrix_result)
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
    #Get the scheduled shuffle list

    res_path=pso.getGBest().getPBest()
    for e_i in range(args.epochs):
        curr_idx = res_path[e_i]
        shuffle_list_sorted[e_i] = shuffle_list[curr_idx] #[nepoch,sample]
        #epoch_list_sharded_sorted[e_i] = epoch_list_sharded[curr_idx] #[nepoch,ngpu,sample_p_gpu]
    #Schedule the node swaps
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
    np.save(os.path.join(args.save_path,'shuffled_list_sorted.npy'), shuffle_list_sorted)
    with open(os.path.join(args.save_path,'idx_to_load_total'), 'wb') as fp:
        pickle.dump(idx_to_load_total,fp)
else:
    shuffle_list = np.load(os.path.join(args.save_path,'original_shuffle_list.npy'))
    shuffle_list_sorted = np.load(os.path.join(args.save_path,'shuffled_list_sorted.npy'))
    with open(os.path.join(args.save_path,'idx_to_load_total'), 'rb') as fp:   # Unpickling
        idx_to_load_total = pickle.load(fp)