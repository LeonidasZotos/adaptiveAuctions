#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=double_check_test
#SBATCH --mem=64GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate
 
adaptive_auction_update_rule=("simple_bandit")
adaptive_auction_action_selection=("e_greedy_exp_decay")
action_selection_hyperparameters=("0 1 0.999" "0 1 0.998" "0 1 0.997" "0 1 0.996" "0 1 0.995" "0 1 0.99" "0 1 0.98" "0 1 0.97" "0 1 0.95" "0 1 0.90" "0 1 0.85")
epochs=10000
num_of_simulations=100

for var1 in "${adaptive_auction_update_rule[@]}"
do
    for var2 in "${adaptive_auction_action_selection[@]}"
    do
        for var3 in "${action_selection_hyperparameters[@]}"
        do
            command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule $var1 --adaptive_auction_action_selection $var2 --action_selection_hyperparameters $var3 --adaptive_auction_discretization 25"
            echo "Executing: $command"
            eval $command
        done
    done
done

deactivate



