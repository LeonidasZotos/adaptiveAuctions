#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=epoch_sweep
#SBATCH --mem=64GB

module purge
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/.envs/cars/bin/activate

num_of_simulations=250
# num_of_epochs=3000
eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs 5000  --results_folder results/epochs_5000 --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --all_cars_bid

eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs 6000  --results_folder results/epochs_6000 --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --all_cars_bid

eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs 7000  --results_folder results/epochs_7000 --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --all_cars_bid

eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs 9000  --results_folder results/epochs_9000 --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --all_cars_bid

eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs 10000  --results_folder results/epochs_10000 --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --all_cars_bid

eval python3 aatc.py run --num_of_simulations $num_of_simulations --num_of_epochs 20000  --results_folder results/epochs_20000 --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection ucb1 --all_cars_bid

deactivate



