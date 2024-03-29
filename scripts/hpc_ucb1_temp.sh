#!/bin/bash
#SBATCH --time=0:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=ucb1_sweep_One
#SBATCH --mem=32GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate

epochs=10000
num_of_simulations=128

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --adaptive_auction_discretization 25"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule svr --adaptive_auction_action_selection ucb1 --adaptive_auction_discretization 25"
echo "Executing: $command"
eval $command

deactivate



