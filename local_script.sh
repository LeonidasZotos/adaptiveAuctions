#!/bin/bash
# This script is most probably used to check if the hpc_script works locally, beforing sending it to the cluster.
adaptive_auction_update_rule=("simple_bandit")
adaptive_auction_action_selection=("random")
action_selection_hyperparameters=("1")
test_boost=("0" "0.25" "0.5" "0.75" "0.999" "1.25" "1.5" "2" "5" "10000")

for var1 in "${adaptive_auction_update_rule[@]}"
do
    for var2 in "${adaptive_auction_action_selection[@]}"
    do
        for var3 in "${action_selection_hyperparameters[@]}"
        do
           for var4 in "${test_boost[@]}"
           do
                # Execute your command with the current parameters
                command="python3 aatc.py run --num_of_simulations 32 --num_of_epochs 5000  --only_winning_bid_moves --adaptive_auction_update_rule $var1 --adaptive_auction_action_selection $var2 --action_selection_hyperparameters $var3 --test_boost $var4"
                echo "Executing: $command"
                eval $command
            done
        done
    done
done