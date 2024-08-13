import numpy as np
import matplotlib.pyplot as plt

# To change some auction parameters, got to line 118 - 123


class AuctionEnvironment:
    def __init__(self, v1, q1, v2, q2, num_bidders=3):
        self.v1 = v1  # Value of item 1
        self.q1 = q1  # Quality of item 1
        self.v2 = v2  # Value of item 2
        self.q2 = q2  # Quality of item 2
        self.num_bidders = num_bidders

    def run_auction(self, bids):
        # Sort bids in descending order and get the indices of the bidders
        sorted_indices = np.argsort(bids)[::-1]
        highest_bidder = sorted_indices[0]
        second_highest_bidder = sorted_indices[1]

        # Assign item 1 to the highest bidder and item 2 to the second-highest bidder
        winning_bids = [bids[highest_bidder], bids[second_highest_bidder]]

        return highest_bidder, second_highest_bidder, winning_bids, bids


class RLBidder:
    def __init__(self, max_valuation, learning_rate=0.1, discount_factor=0.95):
        self.max_valuation = max_valuation  # The maximum possible valuation of the items
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros(self.max_valuation + 1)  # Q-values for each possible bid
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99  # Decay rate for epsilon

    def select_bid(self):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # Explore: choose a random bid
            bid = np.random.randint(0, self.max_valuation + 1)
        else:
            # Exploit: choose the best known bid
            bid = np.argmax(self.q_table)
        return bid

    def update_q_table(self, bid, reward):
        # Update the Q-value for the chosen bid
        self.q_table[bid] += self.learning_rate * (reward - self.q_table[bid])

    def update_epsilon(self):
        # Decay the exploration rate
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)


def calculate_vcg_payment(bids, item, q1, q2):
    # Sort bids in descending order
    sorted_bids = sorted(bids, reverse=True)

    # The payment is the harm done to other bidders, which can be calculated by the VCG algorithm
    if item == 1:
        harm = (sorted_bids[1] * (q1 - q2) + sorted_bids[2] * q2) / q1
    else:
        harm = (sorted_bids[2] * q2) / q2
    return harm


def calculate_reward(v1, v2, q1, q2, bids, item):
    if item == 1:
        payment = calculate_vcg_payment(bids, item, q1, q2)
        return (v1 - payment) * q1
    elif item == 2:
        payment = calculate_vcg_payment(bids, item, q1, q2)
        return (v2 - payment) * q2
    else:
        return 0


def run_simulation(v1=100, v2=60, q1=1, q2=0.5, num_bidders=3, num_rounds=1000):
    num_bidders = max(num_bidders, 3)
    env = AuctionEnvironment(v1, q1, v2, q2, num_bidders)
    bidders = [RLBidder(max_valuation=max(v1, v2)) for _ in range(num_bidders)]

    # Lists to store bids for plotting
    winning_bids1 = []
    winning_bids2 = []

    for round in range(num_rounds):
        # Each bidder selects a bid
        bids = [bidder.select_bid() for bidder in bidders]

        # Run the auction and get the results
        highest_bidder, second_highest_bidder, winning_bids, bids = env.run_auction(bids)

        # Store the bids for this round
        winning_bids1.append(winning_bids[0])
        winning_bids2.append(winning_bids[1])

        # Calculate rewards and update the bidders
        for i in range(num_bidders):
            if i == highest_bidder:
                reward = calculate_reward(v1, v2, q1, q2, bids, item=1)
            elif i == second_highest_bidder:
                reward = calculate_reward(v1, v2, q1, q2, bids, item=2)
            else:
                reward = 0  # No item won, no reward

            bidders[i].update_q_table(bids[i], reward)
            bidders[i].update_epsilon()

        # Optional: Print or track results
        # if round % 100 == 0:
        #     print(f"Round {round}: Bids - {bids}, Winner1 - Bidder {highest_bidder}, Winner2 - Bidder {second_highest_bidder}")

    return winning_bids1, winning_bids2


# Run the simulation
v1 = 100
v2 = 60
q1 = 1
q2 = 0.5
num_bidders = 3  # Please put a number greater than or equal to 3
num_rounds = 1000
winning_bids1, winning_bids2 = run_simulation(v1, v2, q1, q2, num_bidders, num_rounds)

rounds = np.arange(num_rounds)
plt.figure(figsize=(10, 10))

# Plotting the bids for each bidder
plt.subplot(2, 1, 1)
plt.scatter(rounds, winning_bids1, label='Item 1 Bids', color="blue", s=5)
plt.xlabel('Round Number')
plt.ylabel('Amount Bid')
plt.title('Bids Over Rounds - Winning Bids for Item 1')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.scatter(rounds, winning_bids2, label='Item 2 Bids', color="green", s=5)
plt.xlabel('Round Number')
plt.ylabel('Amount Bid')
plt.title('Bids Over Rounds - Winning Bids for Item 2')
plt.grid(True)

plt.tight_layout()
plt.show()
