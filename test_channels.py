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
    assert len(rows) == len(cols) == len(ratings), "维度不匹配"
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



def load_ad_data(file_path):
    data = pd.read_csv(file_path)
    data['ad_index'] = np.arange(len(data))
    return data

def weighted_ad_integration(rec_results, top_n, weights):
    
    final_recs = []
    seen_items = set()
    
    # Process each channel in the order of rec_results (weights order should match this order)
    for ch, weight in zip(rec_results.keys(), weights):
        # Calculate the allocated number of items for this channel
        weight = max(0, min(1, weight))
        allocated_n = int(top_n * weight)
        # If allocation is 0 but there are items available, ensure at least one is taken
        # if allocated_n == 0 and rec_results[ch]:
            # allocated_n = 1
        
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

def shapley_value_allocation(ad_data, sparse_matrix, top_n=100):
    
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
    


    def compute_reward_for_subset(channel_subset):
        
        if not channel_subset:
            return 0, 0  
        rec_results = {}
        uniform_weight = 1.0 / len(channel_subset)
        weights_list = [uniform_weight] * len(channel_subset)
        for ch in channel_subset:
            rec_results[ch] = channel_functions[ch](sparse_matrix, top_n)
        
        final_rec_ids = weighted_ad_integration(rec_results, top_n, weights_list)
        print(rec_results)
        return final_rec_ids

    
    for i, ch in enumerate(channels):
        other_channels = [c for c in channels if c != ch]
        for r in range(len(other_channels) + 1):
            for subset in itertools.combinations(other_channels, r):
                S = list(subset)
                result = compute_reward_for_subset(S)
                print(result)
                print('#############################')


if __name__ == "__main__":
    base_dirs = [
        '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_1',
        '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_2',
        '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_3',
        '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_4',
        '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_5'
    ]

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
    rec_results = {}
    weights_list = [0.1] * 10

    for idx, base_dir in enumerate(base_dirs, start=1):
        for i in range(48):
            print(f"Start data {idx} part {i}")
            csv_file_path = os.path.join(base_dir, f"TSI_{i}.0.csv")
            pkl_file_path = os.path.join(base_dir, f"interaction_sparse_matrix_{i}.pkl")
            
            with open(pkl_file_path, 'rb') as f:
                sparse_matrix = pickle.load(f)
            sparse_matrix = csr_matrix(sparse_matrix)
            ad_data = load_ad_data(csv_file_path)
            shapley_value_allocation(
                ad_data, sparse_matrix
            )

