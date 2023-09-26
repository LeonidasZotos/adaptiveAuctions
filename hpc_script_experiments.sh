#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=demo_experiment
#SBATCH --mem=32GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate

num_of_simulations=2000
num_of_epochs=2000
eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs $num_of_epochs  --results_folder results/simple_bandit --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --all_cars_bid

eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs $num_of_epochs  --results_folder results/svr --adaptive_auction_update_rule svr --adaptive_auction_action_selection ucb1 --all_cars_bid

eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs $num_of_epochs  --results_folder results/random --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection random --all_cars_bid


deactivate



