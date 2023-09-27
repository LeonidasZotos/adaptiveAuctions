#!/bin/bash
# This script is most probably used to check if the hpc_script works locally, before sending it to the cluster.


num_of_simulations=100
eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs 5000  --results_folder results/epochs_5000 --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --all_cars_bid
