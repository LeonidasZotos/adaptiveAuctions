#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=parameter_sweep_reverse_sigmoid_decay
#SBATCH --mem=32GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate
 
adaptive_auction_update_rule=("simple_bandit" "svr")
adaptive_auction_action_selection=("reverse_sigmoid_decay")
action_selection_hyperparameters=("0 0.01" "0 0.05" "0 0.1" "0 0.2" "0 0.3" "0 0.4" "0 0.5" "0 0.6")

for var1 in "${adaptive_auction_update_rule[@]}"
do
    for var2 in "${adaptive_auction_action_selection[@]}"
    do
        # Loop through auction_modifier_type
        for var3 in "${action_selection_hyperparameters[@]}"
        do
            # Execute your command with the current parameters
            command="python3 aatc.py run --num_of_simulations 64 --num_of_epochs 5000  --only_winning_bid_moves --adaptive_auction_update_rule $var1 --adaptive_auction_action_selection $var2 --action_selection_hyperparameters $var3"
            echo "Executing: $command"
            eval $command
        done
    done
done

deactivate