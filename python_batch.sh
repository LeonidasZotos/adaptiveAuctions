#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=spsa_and_static
#SBATCH --mem=4GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate
 
python3 aatc.py simulate --num_of_simulations 10 --grid_size 7 --num_of_epochs 10000 --congestion_rate 0.4 --auction_modifier_type spsa

python3 aatc.py simulate --num_of_simulations 10 --grid_size 7 --num_of_epochs 10000 --congestion_rate 0.4 --auction_modifier_type static

deactivate