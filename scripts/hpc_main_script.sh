#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=adaptive_vs_random
#SBATCH --mem=128GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate

epochs=10000
num_of_simulations=1000

# Execute your command with the current parameters
command="python3 ../aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection boltzmann"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection random"
echo "Executing: $command"
eval $command

deactivate