import numpy as np
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Run the simulation with the desired auction parameters
VALUATIONS = [100] * 2
CONVERGE_WINDOW = 1000
NUM_AUCTIONS = 10000
NUM_EXPERIMENTS = 1000

CONFIDENCE_INTERVALS = True
ALPHA = 0.1
GAMMA = 0.95


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepQLearningBidder:
    def __init__(self, valuation, learning_rate=0.01, discount_factor=0.95, epsilon=1.0):
        self.valuation = valuation
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = []
        self.batch_size = 32
        self.state_size = 1  # Current bid price as state
        self.action_size = valuation + 1  # Possible bids
        
        self.model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def select_bid(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.valuation + 1)
        else:
            state = torch.tensor([[self.valuation]], dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def update_auction_result(self, is_winner, payment, my_bid, winner_bid):
        reward = self._calculate_reward(is_winner, payment, my_bid, winner_bid)
        next_state = my_bid  # New state after auction
        self._store_experience(self.valuation, my_bid, reward, next_state)
        self._update_model()

    def _calculate_reward(self, is_winner, payment, my_bid, winner_bid):
        if is_winner:
            return self.valuation - payment  # Profit
        return 0  # No reward for losing

    def _store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def _update_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32).view(-1, 1)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32).view(-1, 1)
        
        q_values = self.model(states).gather(1, actions.view(-1, 1)).squeeze()
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + self.discount_factor * next_q_values
        
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        

# one exeperiment with multiple auctions, collect the bidder bids and winning bids
class AuctionEnvironment:
    def __init__(self, valuations, auction_type, visibility, num_auctions):
        self.valuations = valuations
        self.bidders_num = len(self.valuations)
        self.auction_type = auction_type  # Auction type: "first-price" or "second-price"
        self.visibility = visibility
        self.num_auctions = num_auctions        

    
    def run_simulation(self):
        global ALPHA, GAMMA
        bidders = [DeepQLearningBidder(self.valuations[i], ALPHA, GAMMA) for i in range(self.bidders_num)]

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
        self.num_auctions = NUM_AUCTIONS
        self.num_experiments = NUM_EXPERIMENTS
        # TODO: hacking about valuations. Use the first one as the standard. It is problemtic if the bidders have different valuations.
        self.expected_value = VALUATIONS[0]/2 if self.auction_type == "first-price" else VALUATIONS[0]

        self.env = AuctionEnvironment(VALUATIONS, self.auction_type, self.visibility, self.num_auctions)

    def run_experiments(self):
        experiments_bidder_bids = []       # experiments_bidder_bids[experiment][auction][bidder]
        experiments_winning_bids = []      # experiments_winning_bids[experiment][auction]

        for i in range(self.num_experiments):
            bidder_bids, winning_bids = self.env.run_simulation()
            experiments_bidder_bids.append(bidder_bids)
            experiments_winning_bids.append(winning_bids)

        avg_winning_bids, std_winning_bids, ci_lower, ci_upper = self._calc_statistics(experiments_bidder_bids, experiments_winning_bids)

        converge_value, t_stat, p_value = self._perform_t_test(avg_winning_bids[-CONVERGE_WINDOW:])

        #self._output_raw_data(experiments_bidder_bids)
        self._output_results(converge_value, t_stat, p_value)

        self._show_figure(avg_winning_bids, std_winning_bids, ci_lower, ci_upper, converge_value)
        
    def _calc_statistics(self, experiments_bidder_bids, experiments_winning_bids):
        '''
        @param experiments_bidder_bids[experiment][auction][bidder]: the raw bidder bid for each experiment, each auctions and each bidder
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

    def _show_figure(self, avg_winning_bids, std_winning_bids, ci_lower, ci_upper, converge_value):
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
        file_name = self._get_file_name_suffix("raw_data")
        with open(file_name, 'w') as file:
            #output header in raw data
            file.write("Experiment,Round,Bidder,Bid_Value\n")

            for exp_id in range(self.num_experiments):
                for auction_id in range(self.num_auctions):
                    for bidder_id, bid in enumerate(experiments_bidder_bids[exp_id][auction_id]):
                        file.write(str(exp_id) + "," + str(auction_id) + ",bidder_" + str(bidder_id) + "," + str(bid) + "\n")

    def _output_results(self, converge_value, t_stat, p_value):
        file_name = self._get_file_name_suffix("result")
        with open(file_name, 'w') as file:
            #output header in raw data
            file.write("converge_value, t_stat, p_value\n")
            file.write(f"{converge_value}, {t_stat}, {p_value}\n")

    def _get_file_name_suffix(self, prefix):
        global ALPHA, GAMMA
        file_name = f"{prefix}_{self.auction_type}_alpha_{ALPHA}_gamma_{GAMMA}.csv"
        return file_name

experiment = Experiments("first-price", "closed")
experiment.run_experiments()

experiment = Experiments("second-price", "closed")
experiment.run_experiments()

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

plt.show()
