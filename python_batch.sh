#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=spsa_and_static
#SBATCH --mem=64GB

module purge
module load Python/3.9.6-GCCcore-11.2.0
 
source $HOME/.envs/cars/bin/activate
 
congestion_rate=("0.1" "0.15" "0.2")
intersection_reward=("time" "time_and_urgency" "valueC")
auction_modifier_type=("spsa" "static")

# Loop through variable1
for var1 in "${congestion_rate[@]}"
do
    # Loop through variable2
    for var2 in "${intersection_reward[@]}"
    do
        # Loop through variable3
        for var3 in "${auction_modifier_type[@]}"
        do
            # Execute your command with the current parameters
            command="python3 aatc.py run --num_of_simulations 10 --grid_size 7 --num_of_epochs 50 --congestion_rate=$var1 --intersection_reward=$var2 --auction_modifier_type=$var3"
            echo "Executing: $command"
            eval $command
        done
    done
done

deactivate