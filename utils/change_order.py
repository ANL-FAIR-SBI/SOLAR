import numpy as np

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

def generate_weight_matrix_cache_fifo_fast(arr,local_batch_size,cache_size,size):
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
                pointer_source=0
                pointer_target=0
                np.sort(source_cache)
                np.sort(curr_samples)
                while pointer_source < cache_size and pointer_target < cache_size:
                    if source_cache[pointer_source] < curr_samples[pointer_target]:
                        pointer_source += 1
                    elif curr_samples[pointer_target] < source_cache[pointer_source]:
                        pointer_target += 1
                    else:
                        cost += 1
                        pointer_source += 1
                        pointer_target += 1
                cost = cache_size - cost
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

#-------------begin of compute
nepochs=100
size=8
LOCAL_BATCH_SIZE = 256
GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * size
cache_size = 641024
#nsamples=6309540 #1.2TB
#nsamples=87632 #17GB
#nsamples=1281167 #imagenet 150GB
#nsamples=50000 #cifair10
nsamples=1281167#imagenet

shuffle_list=np.zeros([nepochs,nsamples])
shuffle_list_sorted=np.zeros([nepochs,nsamples])

#train_data2=ptychoDataset(file_path="/lcrc/project/sbi-fair/baixi/ptychoNN_data/large/h5_data/train/train_t2.h5",rank=rank,idxs=idx_list,cache_size=cache_size)
#train_data3=ptychoDataset(file_path="/lcrc/project/sbi-fair/baixi/ptychoNN_data/large/h5_data/train/train_t2.h5",rank=rank,idxs=idx_list,cache_size=cache_size)
train_number = nsamples
step_size = round(train_number/GLOBAL_BATCH_SIZE)
    
#epoch_list_sharded=np.zeros([nepochs,ngpus,math.ceil(nsamples/ngpus)])
for epoch in range(nepochs):
    idx_arr = np.arange(nsamples)
    np.random.shuffle(idx_arr)
    #idx_arr = MPI.COMM_WORLD.bcast(idx_arr, root=0)
    shuffle_list[epoch] = idx_arr
    #arr_sharded = shard(ngpus, idx_arr)
    #arr_sharded is in the shape of [#gpus, #samples per gpu]
    #epoch_list_sharded[epoch]=arr_sharded

np.save('/path/to/idx/list/shuffle_list_orig.npy', shuffle_list)
idx_to_load_total=[]
mat_start_time = time.perf_counter()
matrix_result = generate_weight_matrix_cache_fifo_new(shuffle_list,LOCAL_BATCH_SIZE,cache_size,size)
mat_time = time.perf_counter() - mat_start_time
print("Cost matrix done! Time: ",mat_time)
np.save('/path/to/idx/list/matrix_result.npy', matrix_result)
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
    #print(matrix_result[i][i+1])
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
np.save('/path/to/idx/list/shuffle_list_sorted.npy', shuffle_list_sorted)
#-------------end 