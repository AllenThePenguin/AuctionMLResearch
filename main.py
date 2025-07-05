import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

# Run the simulation with the desired auction parameters
VALUATIONS = [100] * 2
CONVERGE_WINDOW = 1000
NUM_AUCTIONS = 10000
NUM_EXPERIMENTS = 1000

CONFIDENCE_INTERVALS = True
SAVE_RAW_DATA = False
ALPHA = 0.1
GAMMA = 0.95

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
                reward += 0.1 * (self.valuation - my_bid)  # Slight positive adjustment for winning by a small margin
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
        self.num_bidders = len(self.valuations)
        self.auction_type = auction_type  # Auction type: "first-price" or "second-price"
        self.visibility = visibility
        self.num_auctions = num_auctions        

    # Run N auctions and return the raw bids for each bidder and winning bids
    def run_simulation(self, save_raw_bids = False):
        global ALPHA, GAMMA
        bidders = [RLBidder(self.valuations[i], ALPHA, GAMMA) for i in range(self.num_bidders)]

        # Lists to store bids for plotting
        bidder_bids = []  # bidder_bids[round][bidder] is the bid of the bidder at a specific round
        winning_bids = [] # winning_bids[round] is the winning bid at a specific round

        for round in range(self.num_auctions):
            # Each bidder selects a bid
            bids = [bidder.select_bid() for bidder in bidders]

            # Run the auction and get the results
            winner_id, winning_bid, payment = self._run_auction(bids)

            # Store the bids for this round
            winning_bids.append(winning_bid)

            if save_raw_bids:
                bidder_bids.append([]) # add a new round bids
                for bid in bids:
                    bidder_bids[-1].append(bid)

            # Calculate rewards and update the bidders
            for i, bidder in enumerate(bidders):
                bidder.update_auction_result(i == winner_id, payment, bids[i], winning_bid if self.visibility == "open" else None)

            # Optional: Print or track results
            # if round % 1000 == 0:
            #     print(f"Round {round}: Bids - {bids}, Winner - Bidder {winner}, Payment - {payment}")

        return bidder_bids, winning_bids

    def _run_auction(self, bids):
        # Determine the winner: the highest bid wins
        winning_bid = max(bids)
        winner_id = np.argmax(bids)

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
        return winner_id, winning_bid, payment

# run multiple experiments, and generate the raw log, also do the statistic and show figures
class Experiments:
    figure_index = 0
    
    def __init__(self, auction_type, visibility):
        self.auction_type = auction_type  # Auction type: "first-price" or "second-price"
        self.visibility = visibility  # visibility: "open" or "closed"
        self.num_auctions = NUM_AUCTIONS
        self.num_experiments = NUM_EXPERIMENTS
        self.num_bidders = len(VALUATIONS)
        # TODO: hacking about valuations. Use the first one as the standard. It is problemtic if the bidders have different valuations.
        self.expected_value = VALUATIONS[0]/2 if self.auction_type == "first-price" else VALUATIONS[0]

        self.env = AuctionEnvironment(VALUATIONS, self.auction_type, self.visibility, self.num_auctions)

    def run_experiments(self):
        experiments_bidder_bids = []       # experiments_bidder_bids[experiment][auction][bidder] is the raw bit value
        experiments_winning_bids = []      # experiments_winning_bids[experiment][auction] is the winner's bid

        exp = self._get_experiment_suffix("")
        for i in range(self.num_experiments):
            if i %50 == 0:
                print(f"{exp} : running simulation {i} out of {self.num_experiments}")
                
            bidder_bids, winning_bids = self.env.run_simulation(i ==0 or SAVE_RAW_DATA)
            experiments_bidder_bids.append(bidder_bids)
            experiments_winning_bids.append(winning_bids)

        print(f"{exp} : start calculating statistics")
        avg_winning_bids, std_winning_bids, ci_lower, ci_upper = self._calc_statistics(experiments_winning_bids)

        print(f"{exp} : start performing t_test")
        converge_value, t_stat, p_value = self._perform_t_test(avg_winning_bids[-CONVERGE_WINDOW:])

        if SAVE_RAW_DATA:
            self._output_raw_data(experiments_bidder_bids)
        self._output_results(converge_value, t_stat, p_value)

        print(f"{exp} : start showing figure")
        self._show_winning_bid_curve(avg_winning_bids, std_winning_bids, ci_lower, ci_upper, converge_value)
        
        print(f"{exp} : start showing figure")
        # just use the first bider of the first experiment for the trajectory
        single_bidder_bids = [row[0] for row in experiments_bidder_bids[0]]
        self._show_bid_trajectory(single_bidder_bids) 
        
                
    def _calc_statistics(self, experiments_winning_bids):
        '''
        @param experiments_winning_bids[experiment][auction]: the raw winning bid for each experiment, each auctions

        @return avg_winning_bids[auction], std_winning_bids[auction], ci_lower[auction], ci_upper[auction]
            avg_winning_bids[auction]: the average of the winning bids for each auction
            std_winning_bids[auction]: the standard deviation of the winning bids for each auction
            ci_lower[auction]: the lower bound of the 95% confidence interval for each auction
            ci_upper[auction]: the upper bound of the 95% confidence interval for each auction
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

        if CONFIDENCE_INTERVALS:
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

    def _show_winning_bid_curve(self, avg_winning_bids, std_winning_bids, ci_lower, ci_upper, converge_value):
        print(f"winning bids: {avg_winning_bids[-3]} , {avg_winning_bids[-2]} , {avg_winning_bids[-1]}" ) # print out the last 3 values
        print(f"standard deviation: {std_winning_bids[-3]} , {std_winning_bids[-2]} , {std_winning_bids[-1]}" ) 
    
        # show the figure
        Experiments.figure_index += 1
        # Assuming you have run the simulation and stored the bids in the variables
        rounds = np.arange(self.num_auctions)

        colors = ['blue', 'red', 'yellow', 'green']
        plt.figure(Experiments.figure_index)
        
        # show the winning bid curve 
        plt.subplot(1, 1, 1)
        plt.scatter(rounds, avg_winning_bids, label='Winning Bids', color=colors[1], s=1)
        if CONFIDENCE_INTERVALS:
            plt.fill_between(rounds, ci_lower, ci_upper, color='blue', alpha=0.2, label='Confident Intervals 95%')
        #plt.axhline(y=self.expected_value, color='green', linestyle='--', linewidth=2, label=f"Expected Nash equilibrium ${self.expected_value}")
        plt.axhline(y=converge_value, color='blue', linestyle='--', linewidth=2, label=f"Nash equilibrium ${converge_value:.1f}")
        plt.xlabel('Auction Index')
        plt.ylabel('Mean Winning Bid($)')
        plt.ylim(40, 102)
        plt.title(self.auction_type.upper() + " a: " + str(ALPHA) + " r:" + str(GAMMA))
        plt.legend(loc="right")
        plt.grid(True)

        # show the standard deviation curve
        # plt.subplot(2, 1, 2)
        # plt.scatter(rounds, std_winning_bids, label='Standard Deviation', color=colors[1], s=1)
        # plt.xlabel('Round Number')
        # plt.ylabel('Standard Deviation')
        # plt.ylim(0, 30)
        # plt.grid(True)

        # save the chart into jpg files
        report_file_name = f"report_{self.auction_type.upper()}_a_{ALPHA}_r_{GAMMA}.jpg"
        plt.savefig(report_file_name, format="JPG")

    def _show_bid_trajectory(self, single_bidder_bids):
        '''
        @param single_bidder_bids[auction] is the raw bit value
        '''
        
        # show the figure
        Experiments.figure_index += 1
        # Assuming you have run the simulation and stored the bids in the variables
        rounds = np.arange(self.num_auctions)

        colors = ['blue', 'red', 'yellow', 'green']
        plt.figure(Experiments.figure_index)
        
        # show the winning bid curve 
        plt.subplot(1, 1, 1)

        plt.scatter(rounds, single_bidder_bids, label='Bids', color=colors[0], s=1)

        plt.xlabel('Auction Index')
        plt.ylabel('Bid($)')
        #plt.ylim(40, 102)
        plt.title(self.auction_type.upper() + " a: " + str(ALPHA) + " r:" + str(GAMMA))
        plt.legend(loc="right")
        plt.grid(True)

        # save the chart into jpg files
        report_file_name = f"trajectory_{self.auction_type.upper()}_a_{ALPHA}_r_{GAMMA}.jpg"
        plt.savefig(report_file_name, format="JPG")
        
    def _perform_t_test(self, avg_winning_bids_window ):
        '''
        @param avg_winning_bids_window[auction]: the average of the winning bids for auctions in the converge window
        '''
        # Find the converge value: try the mean value 
        converge_value = sum(avg_winning_bids_window) / len(avg_winning_bids_window)
        t_stat, p_value = ttest_1samp(avg_winning_bids_window, converge_value)
            
        print('\n')
        #print(f"T-statistic: {t_stat}, P-value: {p_value}, bid: {converge_value}")

        if p_value >= 0.05:
            print(self.auction_type.upper(), f"converge to price: {converge_value:.1f}")
            return converge_value, t_stat, p_value
        else:
            print(self.auction_type.upper(), f"Not converge to price: {converge_value:.1f}")
            return -1, t_stat, p_value

    def _output_raw_data(self, experiments_bidder_bids):
        # experiments_bidder_bids[experiment][auction][bidder]
        file_name = self._get_experiment_suffix("raw_data") + ".csv"
        with open(file_name, 'w') as file:
            #output header in raw data
            file.write("Experiment,Round,Bidder,Bid_Value\n")

            for exp_id in range(self.num_experiments):
                for auction_id in range(self.num_auctions):
                    for bidder_id, bid in enumerate(experiments_bidder_bids[exp_id][auction_id]):
                        file.write(str(exp_id) + "," + str(auction_id) + ",bidder_" + str(bidder_id) + "," + str(bid) + "\n")

    def _output_results(self, converge_value, t_stat, p_value):
        file_name = self._get_experiment_suffix("result") + ".csv"
        with open(file_name, 'w') as file:
            #output header in raw data
            file.write("converge_value, t_stat, p_value\n")
            file.write(f"{converge_value}, {t_stat}, {p_value}\n")

    def _get_experiment_suffix(self, prefix):
        global ALPHA, GAMMA
        file_name = f"{prefix}_{self.auction_type}_alpha_{ALPHA}_gamma_{GAMMA}"
        return file_name

    
experiment = Experiments("first-price", "closed")
experiment.run_experiments()

experiment = Experiments("second-price", "closed")
experiment.run_experiments()
plt.show()


# ALPHA = 0.05
# GAMMA = 0.5
# experiment = Experiments("first-price", "closed")
# experiment.run_experiments()
# experiment = Experiments("second-price", "closed")
# experiment.run_experiments()

# ALPHA = 0.05
# GAMMA = 0.8
# experiment = Experiments("first-price", "closed")
# experiment.run_experiments()
# experiment = Experiments("second-price", "closed")
# experiment.run_experiments()

# ALPHA = 0.05
# GAMMA = 0.99
# experiment = Experiments("first-price", "closed")
# experiment.run_experiments()
# experiment = Experiments("second-price", "closed")
# experiment.run_experiments()

# ALPHA = 0.1
# GAMMA = 0.5
# experiment = Experiments("first-price", "closed")
# experiment.run_experiments()
# experiment = Experiments("second-price", "closed")
# experiment.run_experiments()

# ALPHA = 0.1
# GAMMA = 0.8
# experiment = Experiments("first-price", "closed")
# experiment.run_experiments()
# experiment = Experiments("second-price", "closed")
# experiment.run_experiments()

# ALPHA = 0.1
# GAMMA = 0.99
# experiment = Experiments("first-price", "closed")
# experiment.run_experiments()
# experiment = Experiments("second-price", "closed")
# experiment.run_experiments()

# ALPHA = 0.3
# GAMMA = 0.5
# experiment = Experiments("first-price", "closed")
# experiment.run_experiments()
# experiment = Experiments("second-price", "closed")
# experiment.run_experiments()

# ALPHA = 0.3
# GAMMA = 0.8
# experiment = Experiments("first-price", "closed")
# experiment.run_experiments()
# experiment = Experiments("second-price", "closed")
# experiment.run_experiments()

# ALPHA = 0.3
# GAMMA = 0.99
# experiment = Experiments("first-price", "closed")
# experiment.run_experiments()
# experiment = Experiments("second-price", "closed")
# experiment.run_experiments()

# plt.show()