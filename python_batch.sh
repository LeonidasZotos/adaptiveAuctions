#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=spsa_and_static
#SBATCH --mem=64GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate
 
congestion_rate=("0.1" "0.15" "0.2")
intersection_reward_type=("time" "time_and_urgency")
auction_modifier_type=("spsa" "static")

# Loop through congestion_rate
for var1 in "${congestion_rate[@]}"
do
    # Loop through intersection_reward_types
    for var2 in "${intersection_reward_type[@]}"
    do
        # Loop through auction_modifier_type
        for var3 in "${auction_modifier_type[@]}"
        do
            # Execute your command with the current parameters
            command="python3 aatc.py run --num_of_simulations 500 --grid_size 7 --num_of_epochs 2000 --congestion_rate=$var1 --intersection_reward_type=$var2 --auction_modifier_type=$var3"
            echo "Executing: $command"
            eval $command
        done
    done
done

deactivate