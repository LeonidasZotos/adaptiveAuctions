#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=discretisation_boltzmann_sweep
#SBATCH --mem=128GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate
 
discretisation_min=25
discretisation_max=35
epochs=10000
num_of_simulations=256


for var4 in $(eval echo "{$discretisation_min..$discretisation_max}")
do
# Execute your command with the current parameters
command="python3 ../aatc.py run --sweep_mode --num_of_simulations $num_of_simulations --num_of_epochs $epochs --all_cars_bid --with_hotspots --adaptive_auction_update_rule simple_bandit --adaptive_auction_action_selection boltzmann --action_selection_hyperparameters 1 0.001 --adaptive_auction_discretization $var4"
echo "Executing: $command"
eval $command
done


deactivate



