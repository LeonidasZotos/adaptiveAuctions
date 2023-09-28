#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=discretisation_sweep
#SBATCH --mem=32GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate
 
adaptive_auction_update_rule=("simple_bandit")
adaptive_auction_action_selection=("ucb1")
action_selection_hyperparameters=("0 0.05")
discretisation_min=3
discretisation_max=30

for var1 in "${adaptive_auction_update_rule[@]}"
do
    for var2 in "${adaptive_auction_action_selection[@]}"
    do
        for var3 in "${action_selection_hyperparameters[@]}"
        do
           for var4 in $(eval echo "{$discretisation_min..$discretisation_max}")
           do
                # Execute your command with the current parameters
                command="python3 aatc.py run --num_of_simulations 128 --num_of_epochs 5000  --only_winning_bid_moves --adaptive_auction_update_rule $var1 --adaptive_auction_action_selection $var2 --action_selection_hyperparameters $var3 --adaptive_auction_discretization $var4"
                echo "Executing: $command"
                eval $command
            done
        done
    done
done

deactivate



