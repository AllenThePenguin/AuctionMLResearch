import numpy as np
import matplotlib.pyplot as plt


class AuctionEnvironment:
    def __init__(self, valuation, auction_type="first-price", num_bidders=2):
        self.valuation = valuation  # Valuation of the item (same for both bidders)
        self.num_bidders = num_bidders
        self.auction_type = auction_type  # Auction type: "first-price" or "second-price"

    def run_auction(self, bids):
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

        # Return the winner, payment, and bids
        return winner, payment, bids

    def reset(self):
        # No specific state to reset in this simple auction environment
        pass


class RLBidder:
    def __init__(self, valuation, learning_rate=0.1, discount_factor=0.95):
        self.valuation = valuation  # The fixed valuation of the object
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

    def update_q_table(self, bid, reward):
        # Update the Q-value for the chosen bid
        self.q_table[bid] += self.learning_rate * (reward - self.q_table[bid])

    def update_epsilon(self):
        # Decay the exploration rate
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)


def calculate_reward(winner, payment, valuation):
    if winner:
        # Reward is the difference between the valuation and the payment
        reward = valuation - payment
    else:
        # No reward for losing (could also consider a small negative reward)
        reward = 0
    return reward


def run_simulation(valuation, auction_type="first-price", num_rounds=1000):
    env = AuctionEnvironment(valuation, auction_type=auction_type)
    bidders = [RLBidder(valuation), RLBidder(valuation)]

    # Lists to store bids for plotting
    bids_bidder_1 = []
    bids_bidder_2 = []

    for round in range(num_rounds):
        # Each bidder selects a bid
        bids = [bidder.select_bid() for bidder in bidders]

        # Store the bids for this round
        bids_bidder_1.append(bids[0])
        bids_bidder_2.append(bids[1])

        # Run the auction and get the results
        winner, payment, bids = env.run_auction(bids)

        # Calculate rewards and update the bidders
        for i, bidder in enumerate(bidders):
            reward = calculate_reward(i == winner, payment, valuation)
            bidder.update_q_table(bids[i], reward)
            bidder.update_epsilon()

        # Optional: Print or track results
        if round % 100 == 0:
            print(f"Round {round}: Bids - {bids}, Winner - Bidder {winner}, Payment - {payment}")

    return bids_bidder_1, bids_bidder_2


# Run the simulation with the desired auction type
valuation = 100  # Example valuation for both bidders
auction_type = "first-price"  # or "first-price"
bids_bidder_1, bids_bidder_2 = run_simulation(valuation, auction_type, 1000)

# Assuming you have run the simulation and stored the bids in the variables
# bids_bidder_1, bids_bidder_2
num_rounds = len(bids_bidder_1)
rounds = np.arange(num_rounds)

# Create a figure with two subplots (one above the other)
plt.figure(figsize=(10, 10))

# Subplot for Bidder 1
plt.subplot(2, 1, 1)
plt.scatter(rounds, bids_bidder_1, label='Bidder 1 Bids', color='blue')
plt.xlabel('Round Number')
plt.ylabel('Amount Bid')
plt.title('Bids Over Rounds - Bidder 1')
plt.grid(True)

# Subplot for Bidder 2
plt.subplot(2, 1, 2)
plt.scatter(rounds, bids_bidder_2, label='Bidder 2 Bids', color='green')
plt.xlabel('Round Number')
plt.ylabel('Amount Bid')
plt.title('Bids Over Rounds - Bidder 2')
plt.grid(True)

# Show the plots
plt.tight_layout()  # Adjusts spacing between the plots to prevent overlap
plt.show()
