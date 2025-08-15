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
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from surprise import SVD as SurpriseSVD
from surprise import Dataset, Reader
import math, itertools
import os

warnings.filterwarnings("ignore")


##################################################
# Part 1: Recommendation System
##################################################
def sparse_cosine_similarity(sparse_matrix):
    normalized_matrix = normalize(sparse_matrix, axis=1)
    similarity_matrix = normalized_matrix @ normalized_matrix.T
    return similarity_matrix

# 1. Popularity-based Recommendation
def pop_recommendation_sparse(sparse_matrix, top_n=100):
    item_popularity = np.array(sparse_matrix.sum(axis=0)).flatten()
    top_items = np.argsort(-item_popularity)[:top_n]
    return top_items

# 2. ItemKNN (Item-based KNN)
def item_knn_recommendation_sparse(sparse_matrix, top_n=100):
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
def user_knn_recommendation_sparse(sparse_matrix, top_n=100):
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
def bpr_recommendation_sparse(sparse_matrix, top_n=100):
    model = BayesianPersonalizedRanking(factors=50, iterations=100, learning_rate=0.01)
    model.fit(sparse_matrix)

    def recommend(user_id, top_n):
        scores = model.recommend(user_id, sparse_matrix[user_id], N=top_n, filter_already_liked_items=True)
        recommended_items = scores[0]
        return recommended_items

    return recommend(5,top_n)


# 5. SimpleX (Simplified Matrix Factorization)
def simple_x_recommendation_sparse(sparse_matrix, top_n=100):
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

# 6. Random Recommendation
def random_recommendation_sparse(sparse_matrix, top_n=100):
    num_items = sparse_matrix.shape[1]
    top_items = np.random.choice(num_items, top_n, replace=False)
    return top_items


# 7. SVD-based Recommendation
def svd_recommendation_sparse(sparse_matrix, top_n=100):
    svd = TruncatedSVD(n_components=10)
    user_factors = svd.fit_transform(sparse_matrix)  
    item_factors = svd.components_ 

    def recommend(user_id, top_n=top_n):
        user_vector = user_factors[user_id]
        scores = np.dot(user_vector, item_factors)
        user_ratings = sparse_matrix[user_id].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]
        scores[rated_items] = -np.inf
        recommended_items = np.argsort(-scores)[:top_n]
        return recommended_items

    return recommend(5, top_n)


# 8. Global Average-based Recommendation
def global_average_recommendation_sparse(sparse_matrix, top_n=100):
    global_average = sparse_matrix.sum() / sparse_matrix.count_nonzero()
    item_sums = np.array(sparse_matrix.sum(axis=0)).flatten()
    item_nonzeros = np.diff(sparse_matrix.tocsc().indptr)
    item_nonzeros[item_nonzeros == 0] = 1
    item_averages = item_sums / item_nonzeros
    item_averages[np.isnan(item_averages)] = global_average
    top_items = np.argsort(-item_averages)[:top_n]
    return top_items


# 9. Surprise SVD Recommendation
def surprise_svd_recommendation_sparse(sparse_matrix, top_n=100):
    data = csr_matrix(sparse_matrix)
    rows, cols = data.nonzero()
    ratings = data[rows, cols].A1
    assert len(rows) == len(cols) == len(ratings), "Size False"
    df = pd.DataFrame({'user': rows, 'item': cols, 'rating': ratings})
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SurpriseSVD()
    algo.fit(trainset)

    def recommend(user_id, top_n):
        scores = []
        for i in range(sparse_matrix.shape[1]):
            score = algo.predict(user_id, i).est
            scores.append(score)
        scores = np.array(scores)
        top_items = np.argsort(-scores)[:top_n]
        return top_items

    return recommend(5, top_n)
    


# 10. Weighted Popularity-based Recommendation
def weighted_popularity_recommendation_sparse(sparse_matrix, top_n=100):
    item_popularity = np.array(sparse_matrix.sum(axis=0)).flatten()
    item_counts = np.diff(sparse_matrix.tocsc().indptr)
    item_counts[item_counts == 0] = 1
    weighted_popularity = item_popularity / item_counts
    weighted_popularity[np.isnan(weighted_popularity)] = 0
    top_items = np.argsort(-weighted_popularity)[:top_n]
    return top_items



##################################################
# Part 2: Bidding
##################################################
def load_ad_data(file_path):
    data = pd.read_csv(file_path)
    data['ad_index'] = np.arange(len(data))
    return data

# Abid Strategy
class AbidBiddingStrategy:
    def __init__(self, budget=100, name="AbidBiddingStrategy", cpa=1 / 1.5, category=0, exp_tempral_ratio=None, random_seed=None):
        self.budget = budget
        self.remaining_budget = budget
        self.name = name
        self.cpa = cpa
        self.category = category
        self.random_seed = random_seed
        self._set_random_seed()
        self.base_actions = exp_tempral_ratio

    def _set_random_seed(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, pValues, pValueSigmas):
        if self.base_actions is None or len(self.base_actions) != len(pValues):
            base_actions = np.ones(len(pValues))
        else:
            base_actions = self.base_actions
        alpha = base_actions * self.cpa / pValues.mean()
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
    
    def calculate_separate_rate(self, p_values, winning, qualities=None, raise_i=0):
        if qualities is None or len(qualities) != len(p_values):
            qualities = np.ones(len(p_values))
        else:
            qualities = np.array(qualities)
        return np.sum(p_values * np.array(winning) * qualities) + raise_i

    
    def calculate_separate_diver(self, ad_nums, ad_cates, M=100):
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

    '''
    for idx, ads in winning_ads.items():
        print(f"Rank {idx + 1} winning ad: {ads}")
    '''
    
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


##################################################
# Part 3: Ad integration with weights
##################################################

def weighted_ad_integration(rec_results, top_n, weights):
    
    final_recs = []
    seen_items = set()
    
    # Process each channel in the order of rec_results (weights order should match this order)
    for ch, weight in zip(rec_results.keys(), weights):
        # Calculate the allocated number of items for this channel
        weight = max(0, min(1, weight))
        allocated_n = int(top_n * weight)
        
        count = 0
        for item in rec_results[ch]:
            if item not in seen_items:
                final_recs.append(item)
                seen_items.add(item)
                count += 1
            if count >= allocated_n:
                break

    # If the total number of unique items is less than top_n, fill up from all channels in order.
    if len(final_recs) < top_n:
        for ch in rec_results:
            for item in rec_results[ch]:
                if len(final_recs) >= top_n:
                    break
                if item not in seen_items:
                    final_recs.append(item)
                    seen_items.add(item)
            if len(final_recs) >= top_n:
                break

    return final_recs

##################################################
# Part 4: Group Shapley Value Allocation 
##################################################
def shapley_value_allocation(ad_data, sparse_matrix, gamma, bidding_strategy, bidding_env, top_n=100, G=3):
    """
    Perform grouped Shapley value based weight allocation.

    Args:
        ad_data: All advertisement data.
        sparse_matrix: Sparse matrix representing the one - to - one correspondence between users and ads.
        gamma: Parameter.
        bidding_strategy: Pre - designed bidding strategy.
        bidding_env: Pre - designed bidding environment.
        top_n: The number of top recommendations, default is 100.
        G: The number of groups, default is 3.

    Returns:
        normalized_weights: Normalized weights for each channel.
        final_revenue: Final revenue.
        final_score: Final score.
        simulation_results: Simulation results including marginal contributions.
    """

    channels = ['pop', 'item_knn', 'user_knn', 'bpr', 'simple_x', 'random', 'SVD', 'gar', 's_SVD', 'wpr']
    channel_functions = {
        'pop': pop_recommendation_sparse,
        'item_knn': item_knn_recommendation_sparse,
        'user_knn': user_knn_recommendation_sparse,
        'bpr': bpr_recommendation_sparse,
        'simple_x': simple_x_recommendation_sparse,
        'random': random_recommendation_sparse,
        'SVD': svd_recommendation_sparse,
        'gar': global_average_recommendation_sparse,
        's_SVD': surprise_svd_recommendation_sparse,
        'wpr': weighted_popularity_recommendation_sparse
    }
    num_channels = len(channels)

    def channel_grouping():
        """
        Group channels based on overlap similarity.

        Returns:
            groups: A list of groups, where each group is a list of channels.
        """
        # Initialize each channel as a separate group
        groups = [[ch] for ch in channels]
        while len(groups) > G:
            # Each channel group retrieves a list
            group_lists = {}
            for group in groups:
                rec_results = {}
                for ch in group:
                    rec_results[ch] = channel_functions[ch](sparse_matrix, top_n)
                final_rec_ids = set()
                for rec in rec_results.values():
                    final_rec_ids.update(rec)
                group_lists[tuple(group)] = final_rec_ids

            # Find the most similar pair of groups
            max_sim = -1
            best_pair = None
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    g_i = groups[i]
                    g_j = groups[j]
                    set_i = group_lists[tuple(g_i)]
                    set_j = group_lists[tuple(g_j)]
                    sim = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                    if sim > max_sim:
                        max_sim = sim
                        best_pair = (i, j)

            # Merge the groups
            g_a_index, g_b_index = best_pair
            g_a = groups[g_a_index]
            g_b = groups[g_b_index]
            new_group = g_a + g_b
            groups.pop(max(g_a_index, g_b_index))
            groups.pop(min(g_a_index, g_b_index))
            groups.append(new_group)

        return groups
    
    # Compute the rewards(revenue or score) for the group
    def compute_reward_for_subset(group_subset):
        """
        Compute the reward (revenue and score) for a given subset of groups.

        Args:
            group_subset: A list of groups.

        Returns:
            revenue: The revenue of the auction.
            score: The score of the auction.
        """
        if (not group_subset) or (group_subset == [[]]):
            return 0, 0
        print(group_subset)
        rec_results = {}
        all_channels = []
        for group in group_subset:
            all_channels.extend(group)
        uniform_weight = 1.0 / len(all_channels)
        weights_list = [uniform_weight] * len(all_channels)
        for group in group_subset:
            for ch in group:
                rec_results[ch] = channel_functions[ch](sparse_matrix, top_n)

        final_rec_ids = weighted_ad_integration(rec_results, top_n, weights_list)
        recommend_data = ad_data[ad_data['ad_index'].isin(final_rec_ids)].copy()
        sep_data = []
        for group in group_subset:
            for ch in group:
                df = ad_data[ad_data['ad_index'].isin(rec_results[ch])].copy()
                sep_data.append(df)

        revenue, score = run_ad_auction(recommend_data, sep_data, gamma, bidding_strategy, bidding_env)
        return revenue, score

    # Group the channels
    groups = channel_grouping()
    print(groups)
    num_groups = len(groups)

    # Calculate Shapley values for each group
    group_shapley_values = {tuple(group): 0.0 for group in groups}
    simulation_results = []
    factorial_N = math.factorial(num_groups)

    # Update Shapley weights for each group
    for i, group in enumerate(groups):
        other_groups = [g for g in groups if g != group]
        for r in range(len(other_groups) + 1):
            for subset in itertools.combinations(other_groups, r):
                S = list(subset)
                # print(S)
                revenue_S, score_S = compute_reward_for_subset(S)
                # print("Finished")
                S_union = S + [group]
                revenue_S_union, score_S_union = compute_reward_for_subset(S_union)
                delta = score_S_union - score_S
                weight_factor = (math.factorial(len(S)) * math.factorial(num_groups - len(S) - 1)) / factorial_N
                marginal_contribution = weight_factor * delta
                group_shapley_values[tuple(group)] += marginal_contribution

                simulation_results.append({
                    'group': group,
                    'subset': S,
                    'score_S': score_S,
                    'score_S_union': score_S_union,
                    'delta': delta,
                    'weight_factor': weight_factor,
                    'marginal_contribution': marginal_contribution
                })

    # Compute normalized weights for each group
    total_group_phi = sum(group_shapley_values.values())
    if total_group_phi == 0:
        group_normalized_weights = {tuple(group): 1.0 / num_groups for group in groups}
    else:
        group_normalized_weights = {tuple(group): group_shapley_values[tuple(group)] / total_group_phi for group in groups}

    # Allocate weights within each group
    normalized_weights = {}
    for group in groups:
        group_weight = group_normalized_weights[tuple(group)]
        group_channels = len(group)
        if group_channels == 1:
            normalized_weights[group[0]] = group_weight
        else:
            # Use Shapley value allocation within the group
            group_shapley = {ch: 0.0 for ch in group}
            factorial_group = math.factorial(group_channels)
            for i, ch in enumerate(group):
                other_channels = [c for c in group if c != ch]
                for r in range(len(other_channels) + 1):
                    for subset in itertools.combinations(other_channels, r):
                        S = list(subset)
                        revenue_S, score_S = compute_reward_for_subset([S])
                        S_union = S + [ch]
                        revenue_S_union, score_S_union = compute_reward_for_subset([S_union])
                        delta = score_S_union - score_S
                        weight_factor = (math.factorial(len(S)) * math.factorial(group_channels - len(S) - 1)) / factorial_group
                        marginal_contribution = weight_factor * delta
                        group_shapley[ch] += marginal_contribution

            total_group_phi_inner = sum(group_shapley.values())
            if total_group_phi_inner == 0:
                inner_normalized_weights = {ch: 1.0 / group_channels for ch in group}
            else:
                inner_normalized_weights = {ch: group_shapley[ch] / total_group_phi_inner for ch in group}
            for ch in group:
                normalized_weights[ch] = group_weight * inner_normalized_weights[ch]

    # Compute the final revenue and score
    rec_results_final = {}
    for ch in channels:
        rec_results_final[ch] = channel_functions[ch](sparse_matrix, top_n)
    weights_list_final = [normalized_weights[ch] for ch in channels]
    final_rec_ids = weighted_ad_integration(rec_results_final, top_n, weights_list_final)
    recommend_data_final = ad_data[ad_data['ad_index'].isin(final_rec_ids)].copy()
    sep_data_final = []
    for ch in channels:
        df = ad_data[ad_data['ad_index'].isin(rec_results_final[ch])].copy()
        sep_data_final.append(df)

    final_revenue, final_score = run_ad_auction(recommend_data_final, sep_data_final, gamma, bidding_strategy, bidding_env)

    return normalized_weights, final_revenue, final_score, simulation_results



# Multiple channels
if __name__ == "__main__":
    '''
    base_dirs = [
        '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_1',
        '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_2',
        '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_3',
        '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_4',
        '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_5'
    ]
    '''
    base_dirs = [
        '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_1'
    ]
    output_dir = '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/output_5Channels_group'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    top_n = 100
    gamma = 1e-10
    channels = ['pop', 'item_knn', 'user_knn', 'bpr', 'simple_x', 'random', 'SVD', 'gar', 's_SVD', 'wpr']
    channel_functions = {
        'pop': pop_recommendation_sparse,
        'item_knn': item_knn_recommendation_sparse,
        'user_knn': user_knn_recommendation_sparse,
        'bpr': bpr_recommendation_sparse,
        'simple_x': simple_x_recommendation_sparse,
        'random': random_recommendation_sparse,
        'SVD': svd_recommendation_sparse,
        'gar': global_average_recommendation_sparse,
        's_SVD': surprise_svd_recommendation_sparse,
        'wpr': weighted_popularity_recommendation_sparse
    }
    
    revenue_shapley_final = []
    revenue_bench_final = []
    score_shapley_final = []
    score_bench_final = []
    
    for idx, base_dir in enumerate(base_dirs, start=1):
        cumulative_revenue_shapley = 0
        cumulative_revenue_bench = 0
        cumulative_score_shapley = 0
        cumulative_score_bench = 0
        
        file_indices = []
        revenue_shapley_list = []
        revenue_bench_list = []
        score_shapley_list = []
        score_bench_list = []
        
        bidding_strategy = AbidBiddingStrategy(budget=100, random_seed=42)
        bidding_env = BiddingEnvironment(random_seed=42)
        
        for i in range(1):
            csv_file_path = os.path.join(base_dir, f"TSI_{i}.0.csv")
            pkl_file_path = os.path.join(base_dir, f"interaction_sparse_matrix_{i}.pkl")
            
            with open(pkl_file_path, 'rb') as f:
                sparse_matrix = pickle.load(f)
            sparse_matrix = csr_matrix(sparse_matrix)
            ad_data = load_ad_data(csv_file_path)
            
            normalized_weights, final_revenue, final_score, simulation_results = shapley_value_allocation(
                ad_data, sparse_matrix, gamma, bidding_strategy, bidding_env
            )
            
            rec_results_final = {}
            for ch in channels:
                rec_results_final[ch] = channel_functions[ch](sparse_matrix, top_n)
            
            inte_results = weighted_ad_integration(rec_results_final, top_n, [0.1]*10)
            recommend_data_final = ad_data[ad_data['ad_index'].isin(inte_results)].copy()
            
            sep_data_final = []
            for ch in channels:
                df = ad_data[ad_data['ad_index'].isin(rec_results_final[ch])].copy()
                sep_data_final.append(df)
            
            final_revenue_bench, final_score_bench = run_ad_auction(
                recommend_data_final, sep_data_final, gamma, bidding_strategy, bidding_env
            )
            
            cumulative_revenue_shapley += final_revenue
            cumulative_revenue_bench += final_revenue_bench
            cumulative_score_shapley += final_score
            cumulative_score_bench += final_score_bench
            
            file_indices.append(i)
            revenue_shapley_list.append(cumulative_revenue_shapley)
            revenue_bench_list.append(cumulative_revenue_bench)
            score_shapley_list.append(cumulative_score_shapley)
            score_bench_list.append(cumulative_score_bench)
        
        revenue_shapley_final.append(cumulative_revenue_shapley)
        revenue_bench_final.append(cumulative_revenue_bench)
        score_shapley_final.append(cumulative_score_shapley)
        score_bench_final.append(cumulative_score_bench)

        '''
        
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        plt.plot(file_indices, revenue_shapley_list, label='Shapley Revenue', marker='o')
        plt.plot(file_indices, revenue_bench_list, label='Equal Weight Revenue', marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Revenue')
        plt.title(f'Revenue Comparison (Dataset {idx})')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(file_indices, score_shapley_list, label='Shapley Score', marker='o')
        plt.plot(file_indices, score_bench_list, label='Equal Weight Score', marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Score')
        plt.title(f'Score Comparison (Dataset {idx})')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_dataset_{idx}.png'))
        plt.close()
    
    labels = [f'Data_{i}' for i in range(1, 6)]
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, revenue_shapley_final, width, label='Shapley Revenue')
    plt.bar(x + width/2, revenue_bench_final, width, label='Equal Weight Revenue')
    plt.xlabel('Dataset')
    plt.ylabel('Cumulative Revenue')
    plt.title('Cumulative Revenue Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_revenue_comparison.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, score_shapley_final, width, label='Shapley Score')
    plt.bar(x + width/2, score_bench_final, width, label='Equal Weight Score')
    plt.xlabel('Dataset')
    plt.ylabel('Cumulative Score')
    plt.title('Cumulative Score Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_score_comparison.png'))
    plt.close()
    '''

    print(f"Revenue of shapley: {revenue_shapley_final}")
    print(f"Revenue of benchmark: {revenue_bench_final}")
    print(f"Score of shapley: {score_shapley_final}")
    print(f"Score of benchmark: {score_bench_final}")


