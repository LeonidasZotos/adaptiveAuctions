#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=extra_runs
#SBATCH --mem=128GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate
epochs=10000
num_of_simulations=256
# boltzmann
command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection boltzmann --action_selection_hyperparameters 1 0.005 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection boltzmann --action_selection_hyperparameters 1 0.002 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection boltzmann --action_selection_hyperparameters 1 0.001 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection boltzmann --action_selection_hyperparameters 1 0.0005 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

# egreedy decay
command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection e_greedy_decay --action_selection_hyperparameters 1 0.005 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection e_greedy_decay --action_selection_hyperparameters 1 0.002 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection e_greedy_decay --action_selection_hyperparameters 1 0.001 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection e_greedy_decay --action_selection_hyperparameters 1 0.0005 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

# egreedy exp decay
command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection e_greedy_exp_decay --action_selection_hyperparameters 0 1 0.94 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection e_greedy_exp_decay --action_selection_hyperparameters 0 1 0.93 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection e_greedy_exp_decay --action_selection_hyperparameters 0 1 0.92 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection e_greedy_exp_decay --action_selection_hyperparameters 0 1 0.91 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection e_greedy_exp_decay --action_selection_hyperparameters 0 1 0.9 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection e_greedy_exp_decay --action_selection_hyperparameters 0 1 0.85 --adaptive_auction_discretization 23"
echo "Executing: $command"
eval $command

deactivate



