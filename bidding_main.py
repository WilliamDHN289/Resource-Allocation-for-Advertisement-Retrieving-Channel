import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import warnings
import random
import pickle
from collections import Counter
warnings.filterwarnings('ignore')

def load_ad_data(file_path):
    data = pd.read_csv(file_path)
    data['ad_index'] = np.arange(len(data))
    return data

# Abid strategy
class AbidBiddingStrategy:
    def __init__(self, budget=100, name="AbidBiddingStrategy", cpa=1 / 1.5, category=0, exp_tempral_ratio=np.ones(10), random_seed=None):
        self.budget = budget
        self.remaining_budget = budget
        self.base_actions = exp_tempral_ratio
        self.name = name
        self.cpa = cpa
        self.category = category
        self.random_seed = random_seed
        self._set_random_seed()

    def _set_random_seed(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, pValues, pValueSigmas):
        alpha = self.base_actions * self.cpa / pValues.mean()
        bids = alpha * pValues * pValues
        bids[bids < 0] = 0
        return bids

# Bidding Environment (Second Price)
class BiddingEnvironment:
    def __init__(self, reserve_pv_price=0.00001, random_seed=None):
        self.reserve_pv_price = reserve_pv_price
        self.random_seed = random_seed
        self._set_random_seed()

    def _set_random_seed(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

    def simulate_bidding(self, bids, pValues, pValueSigmas):
        sorted_bid_indices = np.argsort(-bids)
        top_bidders = sorted_bid_indices[:5]  # Top 5 winning bidders
        conversions = np.random.binomial(1, pValues[top_bidders])  # Conversion situation (0/1)
        
        # Second price payment
        market_prices = [max(bids[top_bidders[i + 1]], self.reserve_pv_price) for i in range(4)]
        market_prices.append(max(bids[top_bidders[4]], self.reserve_pv_price)) 
        
        return top_bidders, conversions, market_prices

    def calculate_platform_revenue(self, conversions, market_prices):
        # revenue = sum(market_prices[i] if conversions[i] == 1 else 0 for i in range(5))  # Exposing payment
        revenue = sum(market_prices[i] for i in range(5))  # Conversion payment
        return revenue
    
    def calculate_separate_rate(self, p_values, winning, qualities = [1,1,1,1,1,1,1,1,1,1], raise_i = 0):  # Waiting to be updated
        return np.sum(p_values * np.array(winning) * np.array(qualities)) + raise_i
    
    def calculate_separate_diver(self, ad_nums, ad_cates, M=10):
        ad_type_counts = Counter(ad_nums)
        T = len(ad_type_counts)  
        fairness_term = 1 - (sum(t * (t - 1) for t in ad_type_counts.values()) / (M * (M - 1)))
        
        ad_provider_counts = Counter(ad_cates)
        P = len(ad_provider_counts)  
        diversity_term = sum(np.log(p) for p in ad_provider_counts.values())
        
        U = fairness_term + diversity_term
        return U
    
# Bidding process
def run_ad_auction(data, sep_data, gamma, bidding_strategy, bidding_env):
    total_revenue = 0
    winning_ads = {i: [] for i in range(5)}  # Record the winning ads

    p_values = data['pValue'].values
    p_value_sigmas = data['pValueSigma'].values

    # Calculate the bids
    bids = bidding_strategy.bidding(pValues=p_values, pValueSigmas=p_value_sigmas)

    # Simulate the bidding process
    top_bidders, conversions, market_prices = bidding_env.simulate_bidding(bids, p_values, p_value_sigmas)
 
    # Record the winning ads
    for idx, bidder in enumerate(top_bidders):
        winning_ads[idx].append(data.iloc[bidder]['ad_index'])

    # Calculate the total income of the platform
    revenue = bidding_env.calculate_platform_revenue(conversions, market_prices)
    total_revenue += revenue

    for idx, ads in winning_ads.items():
        print(f"Rank {idx + 1} winner id: {ads}")

    
    # Calculate the score of the system
    score = 0

    for channel_retrieve in sep_data:
        p_values_sep = channel_retrieve['pValue'].values
        ad_nums = channel_retrieve['advertiserNumber']
        ad_cates = channel_retrieve['advertiserCategoryIndex']


        winning = [0] * len(channel_retrieve)
        for idx, ad_id in enumerate(channel_retrieve):
            if ad_id in winning_ads:
                winning[idx] = 1

        separate_rate = bidding_env.calculate_separate_rate(p_values_sep, winning)
        separate_diver = bidding_env.calculate_separate_diver(ad_nums, ad_cates)
        
        score += separate_rate + gamma * separate_diver


    return total_revenue, score


if __name__ == "__main__":
    random_seed = 666

    with open('final_recommendations.pkl', 'rb') as f:
        final_recommendations = pickle.load(f)   # Recommended ads from channels
    with open('separate_recommendations.pkl', 'rb') as sf:
        separate_recommendations = pickle.load(sf)  # Recommended ads from each channel
    data = load_ad_data('channel_test.csv')

    print("Shapley")
    print("Ad ids after the combination of channels:", final_recommendations)
    print("Ad ids for each channel:", separate_recommendations)
    print()

    # Select the recommended ads
    recommend_data = data.iloc[final_recommendations]

    # Select the recommended ads for each channel
    sep_recommend_data = []
    for channel_recommendations in separate_recommendations:
        sep_recommend_data.append(data.iloc[channel_recommendations])   

    # Initilize the bidding strategy and environment
    bidding_strategy = AbidBiddingStrategy(budget=100, random_seed=random_seed)
    bidding_env = BiddingEnvironment(random_seed=random_seed)

    # Calculate the total platform revenue
    total_revenue,score = run_ad_auction(recommend_data, sep_recommend_data, 0.001, bidding_strategy, bidding_env)

    print()
    print(f"Total revenue: {total_revenue}")
    print(f"Score of channel group: {score}")






