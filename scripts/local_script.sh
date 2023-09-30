#!/bin/bash
# This script is most probably used to check if the hpc_script works locally, before sending it to the cluster.
 
adaptive_auction_update_rule=("simple_bandit" "svr")
adaptive_auction_action_selection=("boltzmann")
action_selection_hyperparameters=("0 0.01" "0 0.05" "0 0.1" "0 0.2" "0 0.5" "0.5 0.01" "0.5 0.05" "0.5 0.1" "0.5 0.2" "0.5 0.5")
discretisation_min=20
discretisation_max=30
epochs=100
num_of_simulations=10

for var1 in "${adaptive_auction_update_rule[@]}"
do
    for var2 in "${adaptive_auction_action_selection[@]}"
    do
        for var3 in "${action_selection_hyperparameters[@]}"
        do
           for var4 in $(eval echo "{$discretisation_min..$discretisation_max}")
           do
                # Execute your command with the current parameters
                command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule $var1 --adaptive_auction_action_selection $var2 --action_selection_hyperparameters $var3 --adaptive_auction_discretization $var4"
                echo "Executing: $command"
                eval $command
            done
        done
    done
done