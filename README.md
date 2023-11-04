# Adaptive Auctions for Traffic Coordination
Repository for the Master Thesis "Adaptive Auctions for Traffic Coordination"

## Abstract

The focus of this thesis lies on the relatively unexplored topic of using auctions to coordinate traffic in intersections. The primary advantage of this method over the traditional traffic light system is that it takes into account the driver’s urgency, allowing drivers in a hurry to bid more to receive priority. This thesis explores whether there is a benefit in using adaptive, instead of traditional, auctions for the coordination of traffic in a road network. A traffic simulation is built where Adaptive auctions are tasked with assigning priority by balancing the cars’ waiting time and their bids. This design is compared with Random and Zero auctions (random and no consideration of waiting time respectively). Compared to Random and Zero auctions, the current approach led to better traffic dispersion, overall lower waiting times and circumstantially higher trip satisfaction. Furthermore, it was found that there is a tendency to minimise drivers’ waiting time instead of giving priority to agents with high urgency. While these findings highlight the potential of Adaptive auctions, their practicality in a traffic environment is limited and challenging to observe with small sample sizes, despite being statistically significant.

## How To Run

### Requirements
The requirements for this project can be found in the `requirements.txt` file. To install them, run the following command in the terminal:
```bash
pip install -r requirements.txt
```

### Running the Simulation
To run the simulation with the default parameters, simply run:
```bash
python aatc.py run
```
A number of parameters can be changed, as described in the table below:
<details>
  <summary>Optional Parameters</summary>

| Argument Name                         | Default Value | Choices         | Type                 | Help Description                                                                                                                            |
|--------------------------------------|---------------|-----------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| --num_of_epochs                       | 10000         | [1-100000]      | int                  | Number of epochs to run. Defaults to 10000. Must be an integer between 1 and 100000.                                                      |
| --num_of_simulations                  | 50            | [1-10000]       | int                  | Number of simulations to run. Defaults to 50. Must be an integer between 1 and 10000.                                                    |
| --grid_size                           | 3             | [2-9]           | int                  | Size of the traffic grid. Defaults to 3 (9 intersections). Must be an integer between 1 and 9.                                           |
| --queue_capacity                      | 10            | [1-100]         | int                  | Capacity of each car queue. Defaults to 10. Must be an integer between 1 and 100.                                                       |
| --congestion_rate                     | 0.07          | 0.01-1          | float                | Rate of congestion. Defaults to 0.07. Must be a float between 0.01 and 1.                                                                    |
| --with_hotspots                       |               |                 | Boolean              | If enabled, there are periodic hotspots to increase congestion in different intersections.                                                 |
| --wage_time                           | 10            | [1-100]         | int                  | Number of epochs between wage distributions. Defaults to 10. Must be an integer between 1 and 100.                                      |
| --credit_balance                      | 5             | 1-100000        | float                | Initial & Renewal credit balance for each car. Defaults to 5. Must be a float between 1 and 100000.                                     |
| --shared_auction_parameters            |               |                 | Boolean              | All auctions will share parameters, instead of each auction having its own parameters.                                                      |
| --adaptive_auction_action_selection    | e_greedy_exp_decay | [boltzmann, e_greedy_decay, e_greedy_exp_decay, ucb1, reverse_sigmoid_decay, random, zero] | str | Type of auction modifier action selection. Defaults to 'e_greedy_exp_decay'. Must be one of the listed options. |
| --bid_calculation_rule                | linear        | [linear, multiplicative, non-linear] | str | This is the way the bid is calculated. Must be one of 'linear', 'multiplicative', or 'non-linear'.             |
| --adaptive_auction_update_rule        | simple_bandit | [simple_bandit, svr] | str | This is the rule used to update the expected reward for each parameter combination. Defaults to simple_bandit. Must be one of 'simple_bandit' or 'svr'. |
| --auction_episode_length               | 10            | [1-1000]        | int | Length of an episode for the adaptive auction. Defaults to 10. Must be an integer between 1 and 1000.     |
| --action_selection_hyperparameters     |               |                 | float                | Hyperparameters to use for the action selection algorithm (consult documentation for details).             |
| --adaptive_auction_discretization      | 25            | [1-1000]        | int | Number of discrete values to check for each parameter of the adaptive auction. Defaults to 25. Must be an integer between 1 and 100. |
| --only_winning_bid_moves              |               |                 | Boolean              | If enabled, only the car with the winning bid will move.                                                                                    |
| --intersection_reward_type             | mixed_metric_rank | [inact_rank, rank_dist_metric, mixed_metric_rank, mixed_rank_dist_metric] | str | Type of reward for the intersection. Must be one of the listed options. |
| --inact_rank_weight                   | 0.5           | 0-1             | float                | Weight of the inactivity rank used for the intersection auction reward. Defaults to 0.5. Must be a float between 0 and 1. |
| --bid_rank_weight                     | 0.5           | 0-1             | float                | Weight of the bid rank used for the intersection auction reward. Defaults to 0.5. Must be a float between 0 and 1. |
| --all_cars_bid                        |               |                 | Boolean              | If enabled, all cars of a queue can bid, not only the first one.               |
| --shared_bid_generator                |               |                 | Boolean              | If enabled, all bidders will share the same bid generator.                     |
| --bidders_proportion                  | [1, 0, 0, 0, 1] |               | int                  | Proportion for the types of bidders to use. Defaults to [1, 0, 0, 0, 1] (only homogeneous and RL). |
| --bidders_urgency_distribution        | gaussian      | [gaussian, beta] | str | Type of distribution from which the bidders' urgency is sampled. Defaults to 'gaussian'. Must be one of the listed options. |
| --results_folder                      | results/YYYY-MM-DD_HH-MM-SS |  | str | Path to the Results Folder. Defaults to 'results/' + current date & time. |
| --print_grid                          |               |                 | Boolean              | If present, the grid will be printed after each epoch.                        |
| --sweep_mode                         |               |                 | Boolean              | If present, no plots are generated to reduce execution time.                    |
| --low_dpi                            |               |                 | Boolean              | If enabled, all plots are created with low dpi to reduce execution time.         |

</details>


### Cleaning Previous Results
To clean the results of previous simulations (including cache), simply run:
```bash
python aatc.py clean
```
