#!/bin/sh
#SBATCH -J Job_GAN # <-- Change that
#SBATCH -p volta # kepler # pascal # gpu # <-- Could be also teslak40 (less memory)
#SBATCH --mem-per-cpu=5gb
#SBATCH --mem=5gb
#SBATCH --ntasks=1  # Number of cores
#SBATCH --nodes=1  # All cores on one machine
#SBATCH --gres=gpu:1 # <-- Specify that you want to use 1 gpu for your python prog
#SBATCH --gres-flags=enforce-binding # active l’affinité CPU:GPU
#SBATCH -A h146  # <-- Do not change, we are all working under the same project
#SBATCH -t 4-2  # <-- IMPORTANT :  this the duration of the simulation at the format : dd-hh:mm:ss. The simulation stopped even if not finished
#SBATCH -N 1 # <-- The number of node you want to use
#SBATCH -o log_%x_out_%j.log  # <-- the name of the file where the output of the simulation is written
#SBATCH -e log_%x_err_%j.log  # <-- the name of the file where errors of the simulation are written

module purge
module load userspace/all
module load cuda/10.1
module load python3/3.6.3

python3 test.py
