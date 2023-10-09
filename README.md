# adaptiveAuctions
Repository for the Master Thesis "Adaptive Auctions for Traffic Coordination"

## Action Selection Parameter Sweep
The sweep was run with the following parameters. Discretisation was found not to matter too much (any value between 20-30 gives the same results)
(MIN_MAX_DELAY_BOOSTS = [0, 20])
- --num_of_simulations == 256
- --num_of_epochs == 10000
- --all_cars_bid == True
- --with_hotspots == True
- --adaptive_auction_discretization == 23
- --grid_size == 3
- --queue_capacity == 10
- --congestion_rate == 0.07
- --wage_time == 10
- --credit_balance == 5
- --shared_auction_parameters == False
- --bid_calculation_rule == "linear"
- --auction_episode_length == 10
- --only_winning_bid_moves == False
- --intersection_reward_type == "mixed_metric_rank"
- --inact_rank_weight == 0.5
- --bid_rank_weight == 0.5
- --shared_bid_generator == False
- --bidders_proportion == [1, 1] (50% homogeneous, 50% heterogeneous)
- --bidders_urgency_distribution == "gaussian"