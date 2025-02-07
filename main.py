import numpy as np
import matplotlib.pyplot as plt

# Run the simulation with the desired auction parameters
valuations = [100] * 2


class RLBidder:
    def __init__(self, valuation, learning_rate=0.1, discount_factor=0.95):
        self.valuation = valuation
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros(self.valuation + 1)  # Q-values for each possible bid
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99  # Decay rate for epsilon

    def select_bid(self):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # Explore: choose a random bid
            bid = np.random.randint(0, self.valuation + 1)
        else:
            # Exploit: choose the best known bid
            bid = np.argmax(self.q_table)
        return bid

    def update_auction_result(self, is_winner, payment, my_bid, winner_bid):
        reward = self._calculate_reward(is_winner, payment, my_bid, winner_bid)
        self._update_q_table(my_bid, reward)
        self._update_epsilon()
            
    def _calculate_reward(self, is_winner, payment, my_bid, winner_bid):
        if is_winner:
            # Reward is the difference between the valuation and the payment
            reward = self.valuation - payment
        else:
            # No reward for losing (could also consider a small negative reward)
            reward = 0
            
        # If winner_bid is provided (open auction)
        if winner_bid is not None and winner_bid != my_bid:
            if my_bid > winner_bid:
                reward += 0.1 * (valuation - my_bid)  # Slight positive adjustment for winning by a small margin
            else:
                reward -= 0.1 * (winner_bid - my_bid)  # Slight penalty for losing by a large margin

        return reward

    def _update_q_table(self, bid, reward):
        # Update the Q-value for the chosen bid
        future_reward = np.max(self.q_table)  # Assume single-state Q-learning for simplicity
        self.q_table[bid] += self.learning_rate * (
            reward + self.discount_factor * future_reward - self.q_table[bid]
        )

    def _update_epsilon(self):
        # Decay the exploration rate
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        

# one exeperiment with multiple auctions, collect the bidder bids and winning bids
class AuctionEnvironment:
    def __init__(self, valuations, auction_type, visibility, num_auctions):
        self.valuations = valuations
        self.bidders_num = len(self.valuations)
        self.auction_type = auction_type  # Auction type: "first-price" or "second-price"
        self.visibility = visibility
        self.num_auctions = num_auctions        

    
    def run_simulation(self):
        bidders = [RLBidder(self.valuations[i]) for i in range(self.bidders_num)]

        # Lists to store bids for plotting
        bidder_bids = [] # bidder_bids[round][bidder] is the bid of the bidder at a specific round
        winning_bids = []

        for round in range(self.num_auctions):
            # Each bidder selects a bid
            bids = [bidder.select_bid() for bidder in bidders]

            # Run the auction and get the results
            winner, payment = self._run_auction(bids)
            winner_bid = max(bids)

            # Store the bids for this round
            bidder_bids.append([]) # add a new round bids
            for bid in bids:
                bidder_bids[-1].append(bid)
            winning_bids.append(winner_bid)

            # Calculate rewards and update the bidders
            for i, bidder in enumerate(bidders):
                bidder.update_auction_result(i == winner, payment, bids[i], winner_bid if self.visibility == "open" else None)

            # Optional: Print or track results
            # if round % 1000 == 0:
            #     print(f"Round {round}: Bids - {bids}, Winner - Bidder {winner}, Payment - {payment}")

        return bidder_bids, winning_bids
    
    def _run_auction(self, bids):
        # Determine the winner: the highest bid wins
        winning_bid = max(bids)
        winner = np.argmax(bids)

        if self.auction_type == "first-price":
            # In a first-price auction, the winner pays their own bid
            payment = winning_bid
        elif self.auction_type == "second-price":
            # In a second-price auction, the winner pays the second-highest bid
            second_highest_bid = sorted(bids)[-2]  # The second-highest bid
            payment = second_highest_bid
        else:
            raise ValueError("Invalid auction type specified.")

        # Return the winner, payment
        return winner, payment

# run multiple experiments, and generate the raw log, also do the statistic and show figures
class Experiments:
    figure_index = 0
    
    def __init__(self, auction_type, visibility):
        self.auction_type = auction_type  # Auction type: "first-price" or "second-price"
        self.visibility = visibility  # visibility: "open" or "closed"
        self.num_auctions = 7000
        self.num_experiments = 1000
        
        self.env = AuctionEnvironment(valuations, self.auction_type, self.visibility, self.num_auctions)

    def run_experiments(self):
        experiments_bidder_bids = []       # experiments_bidder_bids[experiment][auction][bidder]
        experiments_winning_bids = []      # experiments_winning_bids[experiment][auction]

        for i in range(self.num_experiments):
            bidder_bids, winning_bids = self.env.run_simulation()
            experiments_bidder_bids.append(bidder_bids)
            experiments_winning_bids.append(winning_bids)

        avg_winning_bids, std_winning_bids, ci_lower, ci_upper = self._calc_statistics(experiments_bidder_bids, experiments_winning_bids)

        #self._output_raw_data(experiments_bidder_bids)

        self._show_figure(avg_winning_bids, std_winning_bids, ci_lower, ci_upper)
        
    def _calc_statistics(self, experiments_bidder_bids, experiments_winning_bids):
        '''
        @param experiments_bidder_bids[experiment][auction][bidder]: the raw bidder bid for each experiment, each auctions and each bidder
        @param experiments_winning_bids[experiment][auction]: the raw winning bid for each experiment, each auctions

        @return avg_winning_bids[auction], std_winning_bids[auction], ci_lower[auction], ci_upper[auction]
            avg_winning_bids: the average of the winning bids for each auction
            std_winning_bids: the standard deviation of the winning bids for each auction
            ci_lower: the lower bound of the 95% confidence interval for each auction
            ci_upper: the upper bound of the 95% confidence interval for each auction
        '''

        # calculate the average winning bid from experiments_winning_bids
        avg_winning_bids = [0] * self.num_auctions # avg_winning_bids[auction] 

        for auction_id in range(self.num_auctions):
            for exp_id in range(self.num_experiments):
                avg_winning_bids[auction_id] += experiments_winning_bids[exp_id][auction_id]
    
        for auction_id in range(self.num_auctions):
            avg_winning_bids[auction_id] /= self.num_experiments

        # calculate the Standard Deviation and 95% confidence intervals
        std_winning_bids = [0] * self.num_auctions  # std_winning_bids[auction]
        ci_lower = [0] * self.num_auctions          # Lower bound of the 95% CI
        ci_upper = [0] * self.num_auctions          # Upper bound of the 95% CI

        t_critical = 1.96  # Approximation for 95% confidence interval (large sample size)

        for auction_id in range(self.num_auctions):
            squared_diff_sum = 0
            for exp_id in range(self.num_experiments):
                deviation = experiments_winning_bids[exp_id][auction_id] - avg_winning_bids[auction_id]
                squared_diff_sum += deviation ** 2
            
            # Compute variance and standard deviation
            variance = squared_diff_sum / self.num_experiments
            std_winning_bids[auction_id] = variance ** 0.5

            # Compute Standard Error of the Mean (SEM)
            sem = std_winning_bids[auction_id] / (self.num_experiments ** 0.5)

            # Calculate 95% CI bounds
            ci_lower[auction_id] = avg_winning_bids[auction_id] - t_critical * sem
            ci_upper[auction_id] = avg_winning_bids[auction_id] + t_critical * sem

        return avg_winning_bids, std_winning_bids, ci_lower, ci_upper

    def _show_figure(self, avg_winning_bids, std_winning_bids, ci_lower, ci_upper):
        # show the figure
        Experiments.figure_index += 1
        # Assuming you have run the simulation and stored the bids in the variables
        rounds = np.arange(self.num_auctions)

        colors = ['blue', 'red', 'yellow', 'green']
        plt.figure(Experiments.figure_index)
        
        # show the winning bid curve 
        plt.subplot(2, 1, 1)
        plt.scatter(rounds, avg_winning_bids, label='Winning Bids', color=colors[1], s=1)
        plt.fill_between(rounds, ci_lower, ci_upper, color='blue', alpha=0.2, label='Shading')

        plt.xlabel('Round Number')
        plt.ylabel('Winning Bid')
        plt.title(self.auction_type.upper())
        plt.grid(True)

        # show the standard deviation curve
        plt.subplot(2, 1, 2)
        plt.scatter(rounds, std_winning_bids, label='Winning Bids', color=colors[1], s=1)

        plt.xlabel('Round Number')
        plt.ylabel('Standard Deviation')
        plt.grid(True)

        print(avg_winning_bids[-3:]) # print out the last 3 values

    def _output_raw_data(self, experiments_bidder_bids):
        # experiments_bidder_bids[experiment][auction][bidder]
        file_name = "raw_data_{}.csv".format(self.auction_type)
        with open(file_name, 'w') as file:
            #output header in raw data
            file.write("Experiment,Round,Bidder,Bid_Value\n")

            for exp_id in range(self.num_experiments):
                for auction_id in range(self.num_auctions):
                    for bidder_id, bid in enumerate(experiments_bidder_bids[exp_id][auction_id]):
                        file.write(str(exp_id) + "," + str(auction_id) + ",bidder_" + str(bidder_id) + "," + str(bid) + "\n")





experiment = Experiments("first-price", "closed")
experiment.run_experiments()

experiment = Experiments("second-price", "closed")
experiment.run_experiments()

plt.show()
