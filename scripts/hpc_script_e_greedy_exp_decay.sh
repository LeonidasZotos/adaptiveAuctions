#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=e_greedy_exp_decay_sweep
#SBATCH --mem=128GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate
 
adaptive_auction_update_rule=("simple_bandit" "svr")
adaptive_auction_action_selection=("e_greedy_exp_decay")
action_selection_hyperparameters=("0 1 0.999" "0 1 0.998" "0 1 0.999" "0 1 0.998" "0 1 0.997" "0 1 0.996" "0 1 0.995" "0 1 0.99" "0 1 0.98" "0 1 0.97" "0 1 0.95" "1 1 0.999" "1 1 0.998" "1 1 0.999" "1 1 0.998" "1 1 0.997" "1 1 0.996" "1 1 0.995" "1 1 0.99" "1 1 0.98" "1 1 0.97" "1 1 0.95")
discretisation_min=20
discretisation_max=30
epochs=10000
num_of_simulations=256

for var1 in "${adaptive_auction_update_rule[@]}"
do
    for var2 in "${adaptive_auction_action_selection[@]}"
    do
        for var3 in "${action_selection_hyperparameters[@]}"
        do
        #    for var4 in $(eval echo "{$discretisation_min..$discretisation_max}")
        #    do
            # Execute your command with the current parameters
            command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule $var1 --adaptive_auction_action_selection $var2 --action_selection_hyperparameters $var3 --adaptive_auction_discretization 23"
            echo "Executing: $command"
            eval $command
            # done
        done
    done
done

deactivate



