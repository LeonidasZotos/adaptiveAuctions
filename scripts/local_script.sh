#!/bin/bash
# This script is most probably used to check if the hpc_script works locally, before sending it to the cluster.
adaptive_auction_update_rule=("simple_bandit" "svr")
adaptive_auction_action_selection=("ucb1")
action_selection_hyperparameters=("0" "1")
epochs=1000
num_of_simulations=10

for var1 in "${adaptive_auction_update_rule[@]}"
do
    for var2 in "${adaptive_auction_action_selection[@]}"
    do
        for var3 in "${action_selection_hyperparameters[@]}"
        do
            command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule $var1 --adaptive_auction_action_selection $var2 --action_selection_hyperparameters $var3 --adaptive_auction_discretization 23"
            echo "Executing: $command"
            eval $command
        done
    done
done