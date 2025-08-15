import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from implicit.bpr import BayesianPersonalizedRanking
# from lightgcn import LightGCN
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


##################################################
# Part 1: Recommendation System
##################################################
def sparse_cosine_similarity(sparse_matrix):
    normalized_matrix = normalize(sparse_matrix, axis=1)
    similarity_matrix = normalized_matrix @ normalized_matrix.T
    return similarity_matrix

# 1. Popularity-based Recommendation
def pop_recommendation_sparse(sparse_matrix, top_n=10):
    item_popularity = np.array(sparse_matrix.sum(axis=0)).flatten()
    top_items = np.argsort(-item_popularity)[:top_n]
    return top_items

# 2. ItemKNN (Item-based KNN)
def item_knn_recommendation_sparse(sparse_matrix, top_n=10):
    item_similarity = sparse_cosine_similarity(sparse_matrix.T)  # Transpose for item-item similarity

    def recommend(user_id, top_n=top_n):
        # Get user's rated items (non-zero entries in the sparse matrix row)
        user_ratings = sparse_matrix[user_id]  # This is already a sparse row
        rated_items = user_ratings.indices  # Indices of non-zero entries
        
        # Compute scores by aggregating similarities of rated items
        scores = item_similarity[rated_items].sum(axis=0)  # Sum across rated items
        scores = np.squeeze(np.asarray(scores))  # Convert to dense array
        
        # Set scores of already-rated items to -inf to avoid recommending them
        scores[rated_items] = -np.inf
        
        # Get top N recommended item indices
        recommended_items = np.argsort(-scores)[:top_n]
        return recommended_items

    return recommend(5,top_n)

# 3. UserKNN (User-based KNN)
def user_knn_recommendation_sparse(sparse_matrix, top_n=10):
    user_similarity = sparse_cosine_similarity(sparse_matrix)  # User-item matrix for similarity

    def recommend(user_id, top_n=top_n):
        # Get most similar users based on cosine similarity
        similar_users = np.argsort(-user_similarity[user_id])[:top_n]

        # Aggregate the ratings of similar users (mean over rows of similar users)
        # Convert sparse matrix to CSR if it's not already
        similar_user_ratings = sparse_matrix[similar_users].tocsr()  # Ensure it's in CSR format
        
        # Compute the mean across rows of similar users
        mean_ratings = similar_user_ratings.mean(axis=0).A.flatten()  # Use .A to get a dense array
        
        # Set scores of already-rated items to -inf to avoid recommending them
        user_ratings = sparse_matrix[user_id].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]  # Get indices of rated items
        mean_ratings[rated_items] = -np.inf  # Avoid recommending already-rated items

        # Get top N recommended items
        recommended_items = np.argsort(-mean_ratings)[:top_n]
        return recommended_items

    return recommend(5,top_n)

# 4. BPR (Bayesian Personalized Ranking)
def bpr_recommendation_sparse(sparse_matrix, top_n=10):
    model = BayesianPersonalizedRanking(factors=50, iterations=100, learning_rate=0.01)
    model.fit(sparse_matrix)

    def recommend(user_id, top_n):
        scores = model.recommend(user_id, sparse_matrix[user_id], N=top_n, filter_already_liked_items=True)
        recommended_items = scores[0]
        return recommended_items

    return recommend(5,top_n)


# 5. SimpleX (Simplified Matrix Factorization)
def simple_x_recommendation_sparse(sparse_matrix, top_n=10):
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=10)
    user_factors = nmf.fit_transform(sparse_matrix)
    item_factors = nmf.components_

    def recommend(user_id, top_n):
        user_vector = user_factors[user_id]
        scores = np.dot(user_vector, item_factors)
        recommended_items = np.argsort(-scores)[:top_n]
        return recommended_items

    return recommend(5,top_n)



##################################################
# Part 2: Bidding
##################################################
def load_ad_data(file_path):
    data = pd.read_csv(file_path)
    data['ad_index'] = np.arange(len(data))
    return data

# Abid Strategy
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

# Bidding Environment
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
        conversions = np.random.binomial(1, pValues[top_bidders])  # Conversion (0/1)
        
        market_prices = [max(bids[top_bidders[i + 1]], self.reserve_pv_price) for i in range(4)]
        market_prices.append(max(bids[top_bidders[4]], self.reserve_pv_price)) 
        
        return top_bidders, conversions, market_prices

    def calculate_platform_revenue(self, conversions, market_prices):
        revenue = sum(market_prices[i] for i in range(5))
        return revenue
    
    def calculate_separate_rate(self, p_values, winning, qualities = [1]*10, raise_i = 0):
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

# Bidding Process
def run_ad_auction(recommend_data, sep_data, gamma, bidding_strategy, bidding_env):
    total_revenue = 0
    winning_ads = {i: [] for i in range(5)}  

    p_values = recommend_data['pValue'].values
    p_value_sigmas = recommend_data['pValueSigma'].values

    bids = bidding_strategy.bidding(pValues=p_values, pValueSigmas=p_value_sigmas)
    top_bidders, conversions, market_prices = bidding_env.simulate_bidding(bids, p_values, p_value_sigmas)
    
    for idx, bidder in enumerate(top_bidders):
        winning_ads[idx].append(recommend_data.iloc[bidder]['ad_index'])

    revenue = bidding_env.calculate_platform_revenue(conversions, market_prices)
    total_revenue += revenue

    for idx, ads in winning_ads.items():
        print(f"Rank {idx + 1} winner id: {ads}")
    
    score = 0
    for channel_retrieve in sep_data:
        p_values_sep = channel_retrieve['pValue'].values
        ad_nums = channel_retrieve['advertiserNumber']
        ad_cates = channel_retrieve['advertiserCategoryIndex']

        winning = [0] * len(channel_retrieve)
        for idx, ad_id in enumerate(channel_retrieve['ad_index']):
            for ads in winning_ads.values():
                if ad_id in ads:
                    winning[idx] = 1
                    break

        separate_rate = bidding_env.calculate_separate_rate(p_values_sep, winning)
        separate_diver = bidding_env.calculate_separate_diver(ad_nums, ad_cates)
        score += separate_rate + gamma * separate_diver

    return total_revenue, score


############################################################################
# Part 3: Training MAB of beta distribution with mini-batch 
############################################################################
def train_mab(sparse_matrix, ad_data, top_n=10, batch_size=50, gamma=0.001):
    channels = ['pop', 'item_knn', 'user_knn', 'bpr', 'simple_x']
    # Initialize the parameters of Beta distribution
    alpha_params = {ch: 1.0 for ch in channels}
    beta_params  = {ch: 1.0 for ch in channels}
    
    num_ads = sparse_matrix.shape[1]
    num_batches = int(np.ceil(num_ads / batch_size))
    
    revenue_history = []      # Record revenue using trained (sampled) weights
    equal_revenue_history = []  # Record revenue using equal weights

    # Initialize bidding strategy and environment
    bidding_strategy = AbidBiddingStrategy(budget=100, random_seed=666)
    bidding_env = BiddingEnvironment(random_seed=666)
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx+1)*batch_size, num_ads)
        # For ads: select all users but only a subset of ads (columns)
        batch_sparse = sparse_matrix[:, batch_start:batch_end]
        batch_ad_data = ad_data.iloc[batch_start:batch_end].reset_index(drop=True)
        
        # ----- Step 1: Sample the weight by Beta distribution -----
        sampled_weights = {}
        rng = np.random.default_rng(123)  
        for ch in channels:
            w = rng.beta(alpha_params[ch], beta_params[ch])
            sampled_weights[ch] = w
        total_weight = sum(sampled_weights.values())
        for ch in channels:
            sampled_weights[ch] /= total_weight  # Normalization

        # ----- Step 2: Run the recommendation system -----
        rec_results = {}
        rec_results['pop']       = pop_recommendation_sparse(batch_sparse, top_n)
        rec_results['item_knn']  = item_knn_recommendation_sparse(batch_sparse, top_n)
        rec_results['user_knn']  = user_knn_recommendation_sparse(batch_sparse, top_n)
        rec_results['bpr']       = bpr_recommendation_sparse(batch_sparse, top_n)
        rec_results['simple_x']  = simple_x_recommendation_sparse(batch_sparse, top_n)
        # print(rec_results)

        # ----- Step 3: Generate the final recommendation using sampled weights -----
        final_recs = []
        seen_items = set()
        for ch in channels:
            allocated_n = int(top_n * sampled_weights[ch])
            if allocated_n == 0 and len(rec_results[ch]) > 0:
                allocated_n = 1
            count = 0
            for item in rec_results[ch]:
                if item not in seen_items:
                    final_recs.append(item)
                    seen_items.add(item)
                    count += 1
                if count >= allocated_n:
                    break
        # Fill up the recommendation to top_n if needed
        if len(final_recs) < top_n:
            all_indices = list(range(len(batch_ad_data)))
            additional = [idx for idx in all_indices if idx not in seen_items]
            final_recs.extend(additional[:(top_n - len(final_recs))])
        final_recommend_df = batch_ad_data.iloc[final_recs].copy()
        
        # ----- Compute final recommendation using equal weights -----
        equal_weights = {ch: 1.0/len(channels) for ch in channels}
        final_recs_equal = []
        seen_items_equal = set()
        for ch in channels:
            allocated_n = int(top_n * equal_weights[ch])
            if allocated_n == 0 and len(rec_results[ch]) > 0:
                allocated_n = 1
            count = 0
            for item in rec_results[ch]:
                if item not in seen_items_equal:
                    final_recs_equal.append(item)
                    seen_items_equal.add(item)
                    count += 1
                if count >= allocated_n:
                    break
        if len(final_recs_equal) < top_n:
            all_indices = list(range(len(batch_ad_data)))
            additional = [idx for idx in all_indices if idx not in seen_items_equal]
            final_recs_equal.extend(additional[:(top_n - len(final_recs_equal))])
        final_recommend_df_equal = batch_ad_data.iloc[final_recs_equal].copy()
        
        # ----- Construct sep_data for each channel (for auction scoring) -----
        sep_data = []
        for ch in channels:
            channel_indices = rec_results[ch]
            channel_indices = [idx for idx in channel_indices if idx < len(batch_ad_data)]
            channel_df = batch_ad_data.iloc[channel_indices].copy()
            sep_data.append(channel_df)
        
        # ----- Step 4: Calculate platform revenue and score -----
        # Using sampled (trained) weights
        train_revenue, score = run_ad_auction(final_recommend_df, sep_data, gamma, bidding_strategy, bidding_env)
        revenue_history.append(train_revenue)
        # Using equal weights
        equal_revenue, equal_score = run_ad_auction(final_recommend_df_equal, sep_data, gamma, bidding_strategy, bidding_env)
        equal_revenue_history.append(equal_revenue)
        
        print(f"Batch {batch_idx+1}/{num_batches}:")
        print(f"Trained Weights: {sampled_weights}, Score: {score}, Revenue: {train_revenue}")
        print(f"Equal Weights Revenue: {equal_revenue}")
        
        # Make sure the reward is between 0 and 1
        reward = max(0, min(1, score))
        
        # ----- Step 5: Update the parameters -----
        for ch in channels:
            alpha_params[ch] += reward
            beta_params[ch]  += (1 - reward)
    
    # Calculate the mean weight of each channel E[w] = α / (α + β)
    final_expected_weights = {ch: alpha_params[ch] / (alpha_params[ch] + beta_params[ch]) for ch in channels}
    return final_expected_weights, alpha_params, beta_params, revenue_history, equal_revenue_history

############################################################################
# Part 4: Recommend the final result
############################################################################
def generate_final_recommendations(sparse_matrix, top_n, final_weights):
    channels = ['pop', 'item_knn', 'user_knn', 'bpr', 'simple_x']
    rec_results = {}
    rec_results['pop']       = pop_recommendation_sparse(sparse_matrix, top_n)
    rec_results['item_knn']  = item_knn_recommendation_sparse(sparse_matrix, top_n)
    rec_results['user_knn']  = user_knn_recommendation_sparse(sparse_matrix, top_n)
    rec_results['bpr']       = bpr_recommendation_sparse(sparse_matrix, top_n)
    rec_results['simple_x']  = simple_x_recommendation_sparse(sparse_matrix, top_n)
    
    total_w = sum(final_weights[ch] for ch in channels)
    normalized_weights = {ch: final_weights[ch] / total_w for ch in channels}
    
    final_recs = []
    seen_items = set()
    for ch in channels:
        allocated_n = int(top_n * normalized_weights[ch])
        if allocated_n == 0 and len(rec_results[ch]) > 0:
            allocated_n = 1
        count = 0
        for item in rec_results[ch]:
            if item not in seen_items:
                final_recs.append(item)
                seen_items.add(item)
                count += 1
            if count >= allocated_n:
                break
    return rec_results, final_recs

############################################################################
# Part 5: Main
############################################################################
if __name__ == "__main__":
    pkl_file_path = './interaction_sparse_matrix_1.pkl'
    with open(pkl_file_path, 'rb') as f:
        sparse_matrix = pickle.load(f)
    sparse_matrix = sparse_matrix.tocsr()
    
    ad_data = load_ad_data('channel_test.csv')
    
    top_n = 10
    batch_size = 50
    gamma = 0.01  
    
    # ----- Train MAB model (mini-batch) -----
    final_weights, alpha_params, beta_params, revenue_history, equal_revenue_history = train_mab(sparse_matrix, ad_data, top_n, batch_size, gamma)
    
    print("\nExpected weight for each channels")
    print(final_weights)
    
    # ----- Generate the final recommendations -----
    rec_results, final_recommendations = generate_final_recommendations(sparse_matrix, top_n, final_weights)
    
    # Uncomment below if you want to save the results
    # with open('final_recommendations.pkl', 'wb') as f:
    #     pickle.dump(final_recommendations, f)
    # sep_recom_list = list(rec_results.values())
    # with open('separate_recommendations.pkl', 'wb') as f:
    #     pickle.dump(sep_recom_list, f)
    
    # ----- Plot the revenue moving curve -----
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(revenue_history)+1), revenue_history, marker='o', color='blue', label="Trained Weights Revenue")
    plt.plot(range(1, len(equal_revenue_history)+1), equal_revenue_history, marker='x', color='red', label="Equal Weights Revenue")
    plt.xlabel("Mini-batch turn")
    plt.ylabel("Total platform revenue")
    plt.title("Revenue Moving Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("revenue_history.png")
    plt.show()
