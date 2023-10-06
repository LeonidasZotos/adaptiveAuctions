#!/bin/bash
# This script is most probably used to check if the hpc_script works locally, before sending it to the cluster.
 
epochs=300
num_of_simulations=10
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
