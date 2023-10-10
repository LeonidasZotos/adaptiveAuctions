#!/bin/bash
# This script is most probably used to check if the hpc_script works locally, before sending it to the cluster.
epochs=10000
num_of_simulations=50

# Execute your command with the current parameters
command="python3 ../aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection boltzmann"
echo "Executing: $command"
eval $command

command="python3 ../aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection random"
echo "Executing: $command"
eval $command


command="python3 ../aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection zero"
echo "Executing: $command"
eval $command