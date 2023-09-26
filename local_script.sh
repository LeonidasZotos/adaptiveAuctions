#!/bin/bash
# This script is most probably used to check if the hpc_script works locally, before sending it to the cluster.

num_of_simulations=30
num_of_epochs=2000
eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs $num_of_epochs  --results_folder results/simple_bandit --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --all_cars_bid

eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs $num_of_epochs  --results_folder results/svr --adaptive_auction_update_rule svr --adaptive_auction_action_selection ucb1 --all_cars_bid

eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs $num_of_epochs  --results_folder results/random --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection random --all_cars_bid
