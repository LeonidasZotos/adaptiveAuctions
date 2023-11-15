In the playground folder, a number of simulations are developed. `01replication_pardoe2006.py` replicates the experiment by Pardoe et al, 2006. The remainder of the simulations incrementally make changes in their approach to the thesis simulation. The simulations are numbered in the order they were developed.

The  table below presents the simulations along with the parameter that is changed. 

| Experiment                            | Auction Type   | Training on:  | Adaptive Parameter  | Number of Bidders | Bidders have different parameter values | Bidder population characteristics          | Necessary additional changes from previous                 | Notes/Results                                             |
|--------------------------------------|----------------|---------------|---------------------|-------------------|----------------------------------------|-----------------------------------------------|----------------------------------------------------------|--------------------------------------------------------|
| 1) Replication, Pardoe 2006          | English Auction | Revenue       | Reserve Price      | 2                 | False                                  | For each auction, different bidder population, same adaptive auction mechanism | ———                                                      | Results are replicated                                    |
| 2) Different aversions               | English Auction | Revenue       | Reserve Price      | 2                 | True                                   | For each auction, different bidder population, same adaptive auction mechanism | None                                                    | Same/Similar results: Adaptive learning helps              |
| 3) Variable number of bidders        | English Auction | Revenue       | Reserve Price      | Variable, 2-4     | True                                   | For each auction, different bidder population, same adaptive auction mechanism | The bidding strategy is different as the equilibrium doesn’t hold for more than 2 bidders. | Performance increase is not as strong, but definitely still there compared to using random reserve prices. |
| 4) Repeating Auctions                | English Auction | Revenue       | Reserve Price      | Variable, 2-4     | True                                   | Each population is used for 20 auctions, instead of changing after every auction | None                                                    | Same: Performance increase. Learning is not as consistent (jumps every 20 auctions) |
| 5) Measuring max_time_waited         | English Auction | Revenue       | Reserve Price      | Variable, 2-4     | True                                   | Each population is used for 20 auctions, instead of changing after every auction | None                                                    | Same: Here, the experiment is the same, but we also measure max_time_waited. There is no decrease in max_time_waited, as the system doesn’t try to improve in that aspect. |
| 6) Training on max_time_waited       | English Auction | Max_time_waited | Reserve Price      | Variable, 2-4     | True                                   | Each population is used for 20 auctions, instead of changing after every auction | None                                                    | Adapting the reserve price doesn’t lead to a better max_time_waited. The revenue also doesn’t improve in this case, as the reserve is not attempting to increase it. Conclusion: Reserve price does not have a clear effect on max_time_waited. |
| 7) No reserve price                  | English Auction | No training, fixed reserve price at 0 | None: Reserve Price is set to 0 | Variable, 2-4     | True | Each population is used for 20 auctions, instead of changing after every auction | None | A reserve price of 0 performed better in almost all experiments, so setting it to 0 seems appropriate, instead of having to go through the learning to end up with 0 anyway. Revenue is better than using random reserve prices. |
| 8) Only valuation. No loss-aversion | English Auction | No training, fixed reserve price at 0 | None: Reserve Price is set to 0 | Variable, 2-4 | True | Each population is used for 20 auctions, instead of changing after every auction. Agents only have valuation; no loss aversion | None | When the reserve is 0, loss-aversion doesn’t matter. This experiment shows that. The performance is the same as experiment 7 |
| 9) Dynamic/Online population         | English Auction | No training, fixed reserve price at 0 | None: Reserve Price is set to 0 | Variable, 2-4 | True | There is no longer a set of 20 auctions. After each auction, the winner is potentially replaced with new bidders (up to 4). Each bidder is drawn from a unique population | None | Nothing too interesting. Reward is lowered over time as there is no prioritisation based on time waited. |
| 10) Boost parameter                  | English Auction | inact_rank    | Inactivity Boost: boosts the bids depending on the time since last win. Reserve price is set to 0 | Variable, 2-4 | True | After each auction, the winner is potentially replaced with new bidders (up to 4). Each bidder is drawn from a unique population | None | As expected, the algorithm learns to consider the inactivity time the most, ignoring the actual bid. Looking at the bandit valuations, boosts ≥ 4 are equally good. 200 epochs are enough to reach optimum. |
| 11) Mixed Metric                    | English Auction | inact_rank and val_rank (Bid rank + inact_rank)/2 | Inactivity Boost: boosts the bids depending on the time since last win. Reserve price is set to 0 | Variable, 2-4 | True | After each auction, the winner is potentially replaced with new bidders (up to 4). Each bidder is drawn from a unique population | Temperature changes to improve learning a bit. Exploitation is prioritised more. | Central question here is: How much should we consider inactivity, if both the valuation and inactivity_time matter equally? What boost parameter allows us to take inactivity seriously enough, but not too much? Findings: The algorithm learns the correct evaluations, but it takes time as the differences are smaller. Next up: finding a better learning algorithm |