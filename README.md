## SOLAR source code Version 1.0

## Requirments

*1.anaconda
*2.Python 3.8.10
*3.torchvision 0.11.1
*4.h5py 3.7.0 based on hdf5 1.12.0
*5.cudatoolkit 11.3.1
*6.cupy-cuda114 10.0.0
*7.mpi4py 3.1.3
*8.openmpi 4.0.5

## STEP 1: Offline Scheduling

### Generate the shuffled list first

### For example, CDI data

```sh
    $ save_path=/path/to/save/results
```

```sh
    $ python3 ./utils/solar_shuffle.py --size 16 --nnodes 16 --epochs 100 --cache_size 160 --ntrain 87632 --save_path \${save_path}
```

## STEP 2: Launch Training
### Run-time scheduling will be handled in dataloader
### For example, running on a 4 node, 4 gpus per node case on thetagpu

1. Setup anaconda environment
2. 
```sh
    $ source /lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/setup.sh
```

2. Activate conda environment (my_env is copied from base environment and install required packages using conda)
 
```sh
    $ conda activate my_env
```

3. Setup hdf5 for the dataset (h5py is just an api which is build on this hdf5)
```sh
    $ module load hdf5/1.12.0
```

4. Run training code

```sh
    $ mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np 16 -npernode 4 --hostfile \${COBALT_NODEFILE} python3 ./PtychoNN/train.py --device gpu --ngpus 16 --batch_size 256 --cache_size 87632 --epochs 10 --nnodes 4 --gpu_pernode 4 --save_path \${save_path} --dtest_path /path/to/test/dataset/ --dtrain_path /path/to/training/dataset/h5 --root_path /path/to/save/models/and/logs
```

