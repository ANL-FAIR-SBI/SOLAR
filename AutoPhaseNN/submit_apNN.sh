#!/bin/bash
#COBALT -n 1
#COBALT -t 04:00:00 -q full-node
#COBALT -A sbi-fair -O /home/sunbaixi/DemoTest/sdl_ai_workshop-master/01_distributedDeepLearning/Horovod/bx_results/$jobid.hvd_modified
#COBALT --attrs filesystems=home,theta-fs0,grand

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules
#echo $CUDA_VISIBLE_DEVICES > /home/sunbaixi/DemoTest/PtychoNN-master/TF2/visible.txt
source /lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/setup.sh
conda activate dist

for g in 4 8
do
    mpirun -np $g python3 -u /home/sunbaixi/DemoTest/PtychoNN-master/TF2/aurophaseNN/apNN_baseline.py >& /home/sunbaixi/DemoTest/PtychoNN-master/TF2/aurophaseNN/results/baseline_${g}proc.out
done
