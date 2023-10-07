#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=extra_runs_ucb1
#SBATCH --mem=128GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate
epochs=10000
num_of_simulations=256

# egreedy decay
command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --action_selection_hyperparameters 1 0.005 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --action_selection_hyperparameters 1 0.002 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --action_selection_hyperparameters 1 0.001 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --action_selection_hyperparameters 1 0.0005 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

deactivate



