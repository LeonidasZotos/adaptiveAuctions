#!/bin/sh

# Cars pyenv environment needs to be activated beforehand.

# Retrieve all paramters
GRID_SIZE="$1"
QUEUE_CAPACITY="$2"
CONGESTION_RATE="$3"
CREDIT_BALANCE="$4"
WAGE_TIME="$5"
NUM_EPOCHS="20" # For testing, this many epochs is enough.

pwd

python ./aatc.py simulate --grid_size "$GRID_SIZE" --queue_capacity "$QUEUE_CAPACITY" --congestion_rate "$CONGESTION_RATE" --credit_balance "$CREDIT_BALANCE" --wage_time "$WAGE_TIME" --num_of_epochs "$NUM_EPOCHS"