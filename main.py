import numpy as np
import matplotlib.pyplot as plt

# To change some auction parameters, got to line 117 - 121


class AuctionEnvironment:
    def __init__(self, valuation, auction_type="first-price", num_bidders=2):
        self.valuation = valuation  # Valuation of the item
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

    def update_q_table(self, bid, reward, opponent_bid):
        # If opponent_bid is provided (open auction)
        if opponent_bid is not None:
            if bid > opponent_bid:
                reward += 0.1 * (self.valuation - bid)  # Slight positive adjustment for winning by a small margin
            else:
                reward -= 0.1 * (opponent_bid - bid)  # Slight penalty for losing by a large margin

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


def run_simulation(valuation1=100, valuation2=100, auction_type="first-price", visibility="closed", num_rounds=1000):
    env = AuctionEnvironment(max(valuation1, valuation2), auction_type=auction_type)
    bidders = [RLBidder(valuation1), RLBidder(valuation2)]

    # Lists to store bids for plotting
    bids_bidder_1 = []
    bids_bidder_2 = []
    winning_bids = []

    for round in range(num_rounds):
        # Each bidder selects a bid
        bids = [bidder.select_bid() for bidder in bidders]

        # Run the auction and get the results
        winner, payment, bids = env.run_auction(bids)

        # Store the bids for this round
        bids_bidder_1.append(bids[0])
        bids_bidder_2.append(bids[1])
        winning_bids.append(max(bids))

        # Calculate rewards and update the bidders
        for i, bidder in enumerate(bidders):
            if i == 0:
                reward = calculate_reward(i == winner, payment, valuation1)
                opponent_bid = bids[1] if visibility == "open" else None
            else:
                reward = calculate_reward(i == winner, payment, valuation2)
                opponent_bid = bids[0] if visibility == "open" else None
            bidder.update_q_table(bids[i], reward, opponent_bid)
            bidder.update_epsilon()

        # Optional: Print or track results
        # if round % 1000 == 0:
        #     print(f"Round {round}: Bids - {bids}, Winner - Bidder {winner}, Payment - {payment}")

    return bids_bidder_1, bids_bidder_2, winning_bids


# Run the simulation with the desired auction parameters
valuation1 = 100
valuation2 = 100
auction_type = "second-price"
visibility = "open"
num_rounds = 1000
bids_bidder_1, bids_bidder_2, winning_bids = run_simulation(valuation1, valuation2, auction_type, visibility, num_rounds)

# Assuming you have run the simulation and stored the bids in the variables
# bids_bidder_1, bids_bidder_2
rounds = np.arange(num_rounds)

# Create a figure with two subplots (one above the other)
plt.figure(figsize=(10, 10))

# Subplot for Bidder 1
plt.subplot(3, 1, 1)
plt.scatter(rounds, bids_bidder_1, label='Bidder 1 Bids', color='blue', s=5)
plt.xlabel('Round Number')
plt.ylabel('Amount Bid')
plt.title('Bids Over Rounds - Bidder 1')
plt.grid(True)

# Subplot for Bidder 2
plt.subplot(3, 1, 2)
plt.scatter(rounds, bids_bidder_2, label='Bidder 2 Bids', color="green", s=5)
plt.xlabel('Round Number')
plt.ylabel('Amount Bid')
plt.title('Bids Over Rounds - Bidder 2')
plt.grid(True)

# Subplot for Winning Bids
plt.subplot(3, 1, 3)
plt.scatter(rounds, winning_bids, label='Bidder 2 Bids', color="red", s=5)
plt.xlabel('Round Number')
plt.ylabel('Amount Bid')
plt.title('Bids Over Rounds - Winning Bids')
plt.grid(True)

# Show the plots
plt.tight_layout()  # Adjusts spacing between the plots to prevent overlap
plt.show()
