#!/bin/bash
# This script is most probably used to check if the hpc_script works locally, before sending it to the cluster.
 discretisation_min=25
discretisation_max=28
epochs=50
num_of_simulations=10


for var4 in $(eval echo "{$discretisation_min..$discretisation_max}")
do
# Execute your command with the current parameters
command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection boltzmann --action_selection_hyperparameters 1 0.001 --adaptive_auction_discretization $var4"
echo "Executing: $command"
eval $command
done