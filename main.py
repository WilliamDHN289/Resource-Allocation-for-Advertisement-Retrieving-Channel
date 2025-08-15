import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from implicit.bpr import BayesianPersonalizedRanking
# from lightgcn import LightGCN
from sklearn.preprocessing import normalize
import pickle
import warnings

warnings.filterwarnings("ignore")

def sparse_cosine_similarity(sparse_matrix):
    # Normalize the sparse matrix row-wise
    normalized_matrix = normalize(sparse_matrix, axis=1)
    
    # Compute cosine similarity using sparse matrix multiplication
    similarity_matrix = normalized_matrix @ normalized_matrix.T
    
    return similarity_matrix

# 2. Popularity-based Recommendation
def pop_recommendation_sparse(sparse_matrix, top_n=10):
    item_popularity = np.array(sparse_matrix.sum(axis=0)).flatten()
    top_items = np.argsort(-item_popularity)[:top_n]
    return top_items

# 3. ItemKNN (Item-based KNN)
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

# 4. UserKNN (User-based KNN)
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

# 5. BPR (Bayesian Personalized Ranking)
def bpr_recommendation_sparse(sparse_matrix, top_n=10):
    model = BayesianPersonalizedRanking(factors=50, iterations=100, learning_rate=0.01)
    model.fit(sparse_matrix)

    def recommend(user_id, top_n):
        scores = model.recommend(user_id, sparse_matrix[user_id], N=top_n, filter_already_liked_items=True)
        recommended_items = scores[0]
        return recommended_items

    return recommend(5,top_n)

# 6. NeuMF (Neural Matrix Factorization) - PyTorch Implementation
class NeuMF(nn.Module):
    # Define the architecture for NeuMF (e.g., embedding layers, etc.)
    def __init__(self, n_users, n_items, latent_dim):
        super(NeuMF, self).__init__()
        self.user_embedding = nn.Embedding(n_users, latent_dim, sparse=True)
        self.item_embedding = nn.Embedding(n_items, latent_dim, sparse=True)
    
    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        return torch.sum(user_emb * item_emb, dim=1)

def neu_mf_recommendation_sparse(sparse_matrix, top_n=10, latent_dim=10):
    n_users, n_items = sparse_matrix.shape
    # print(n_users, n_items)
    model = NeuMF(n_users, n_items, latent_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SparseAdam(model.parameters(), lr=0.001)
    
    users, items = sparse_matrix.nonzero()
    # print(users.shape, items.shape)
    ratings = torch.FloatTensor(sparse_matrix.data)  # Use sparse_matrix.data directly
    # print(ratings.shape)

    users = torch.LongTensor(users)
    items = torch.LongTensor(items)
    
    # Training
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(users, items)
        
        # print(outputs.shape)
        # print(ratings.shape)
        # outputs = outputs.flatten()
        # ratings = ratings.flatten()

        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()

    def recommend(user_id, top_n):
        user_vector = torch.LongTensor([user_id] * n_items)
        items_vector = torch.LongTensor(range(n_items))
        scores = model(user_vector, items_vector)
        recommended_items = torch.argsort(scores, descending=True)[:top_n]
        return recommended_items.numpy()

    return recommend(5,top_n)

# 7. SimpleX (Simplified Matrix Factorization)
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

'''
# 8. LightGCN (Light Graph Convolutional Network)
def lightgcn_recommendation_sparse(sparse_matrix, top_n=10):
    model = LightGCN(sparse_matrix, n_layers=3, n_epochs=100, lr=0.01)
    model.fit()

    def recommend(user_id, top_n):
        scores = model.predict(user_id)
        recommended_items = np.argsort(-scores)[:top_n]
        return recommended_items

    return recommend(5,top_n)
'''



################################### Main Function ###################################

def run_recommendation_systems(sparse_matrix, top_n=10, group=3):
    # Run each recommendation system and store the results
    recommendation_results = {}
    recommendation_results['pop'] = pop_recommendation_sparse(sparse_matrix, top_n)
    recommendation_results['item_knn'] = item_knn_recommendation_sparse(sparse_matrix, top_n)
    recommendation_results['user_knn'] = user_knn_recommendation_sparse(sparse_matrix, top_n)
    recommendation_results['bpr'] = bpr_recommendation_sparse(sparse_matrix, top_n)
    recommendation_results['simple_x'] = simple_x_recommendation_sparse(sparse_matrix, top_n)
  
    # ----- Step 1: Divide the channels into different groups by similarity -----
    # Initialization: Each channel is divided into a group
    groups = []
    for key, rec_list in recommendation_results.items():
        groups.append({'channels': [key], 'items': set(rec_list)})
    
    # Definition of jaccard similarity
    def jaccard_similarity(set1, set2):
        union = set1.union(set2)
        if len(union) == 0:
            return 0
        return len(set1.intersection(set2)) / len(union)
    
    # Combine the most similarity two groups until reach the target number of groups
    while len(groups) > group:
        max_sim = -1
        pair_to_merge = (None, None)
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                sim = jaccard_similarity(groups[i]['items'], groups[j]['items'])
                if sim > max_sim:
                    max_sim = sim
                    pair_to_merge = (i, j)
       
        i, j = pair_to_merge
        new_channels = groups[i]['channels'] + groups[j]['channels']
        new_items = groups[i]['items'].union(groups[j]['items'])
        new_group = {'channels': new_channels, 'items': new_items}
    
        groups.pop(j)
        groups.pop(i)
        groups.append(new_group)
    
    # ----- Step 2: Generate the weight for each group (Beta distribution) -----
    # Initialization: Beta(1,1)
    alpha_param = 1
    beta_param = 1
    group_weights = []
    for _ in range(len(groups)):
        w = np.random.beta(alpha_param, beta_param)
        group_weights.append(w)
    total_group_weight = sum(group_weights)
    normalized_group_weights = [w / total_group_weight for w in group_weights]
    
    # ----- Step 3: Combine the final result by weight -----
    final_list = []
    seen_items = set() 
    for idx, grp in enumerate(groups):
        allocated_n = int(top_n * normalized_group_weights[idx])
        if allocated_n == 0 and len(grp['items']) > 0:
            allocated_n = 1
        count = 0
        for item in grp['items']:
            if item not in seen_items:
                final_list.append(item)
                seen_items.add(item)
                count += 1
            if count >= allocated_n:
                break

    return recommendation_results, final_list, groups, normalized_group_weights

if __name__ == "__main__":
    pkl_file_path = './interaction_sparse_matrix.pkl' 
    # Load the sparse matrix from the pickle file
    with open(pkl_file_path, 'rb') as f:
        sparse_matrix = pickle.load(f)
    sparse_matrix = sparse_matrix.tocsr()

    top_n = 10  
    separate_recommendations, final_recommendations, groups, group_weights = run_recommendation_systems(sparse_matrix, top_n, group=3)
    
    print("各通道的推荐结果:")
    print(separate_recommendations)
    
    print("\n分组信息 (每组包含的通路及对应推荐物品集合):")
    for grp in groups:
        print(grp)
    
    print("\n归一化后的组权重:")
    print(group_weights)
    
    print("\n整合后的加权推荐结果:")
    print(final_recommendations)
    

    '''
    with open('final_recommendations.pkl', 'wb') as f:
        pickle.dump(final_recommendations, f)
    '''
    
    # 保存各通道的推荐结果
    '''
    sep_recom_list = list(separate_recommendations.values())
    with open('separate_recommendations.pkl', 'wb') as f:
        pickle.dump(sep_recom_list, f)
    '''





'''
########################################## main function ##########################################
def run_recommendation_systems(pkl_file_path, weights, top_n=10):
    # Load the sparse matrix from the pickle file
    with open(pkl_file_path, 'rb') as f:
        sparse_matrix = pickle.load(f)

    sparse_matrix = sparse_matrix.tocsr()

    # Run each recommendation system and store the results
    recommendation_results = {}
    recommendation_results['pop'] = pop_recommendation_sparse(sparse_matrix, top_n)
    recommendation_results['item_knn'] = item_knn_recommendation_sparse(sparse_matrix, top_n)
    recommendation_results['user_knn'] = user_knn_recommendation_sparse(sparse_matrix, top_n)
    recommendation_results['bpr'] = bpr_recommendation_sparse(sparse_matrix, top_n)
    # recommendation_results['neu_mf'] = neu_mf_recommendation_sparse(sparse_matrix, top_n)
    recommendation_results['simple_x'] = simple_x_recommendation_sparse(sparse_matrix, top_n)
    # recommendation_results['lightgcn'] = lightgcn_recommendation_sparse(sparse_matrix, top_n)
  
    # Integrate results with weights
    final_list = []

    total_weight = sum(weights.values())
    seen_items = set()  # To track which items have already been added to final_list
    
    for key, recommended_items in recommendation_results.items():
        weight = weights[key] / total_weight  # Normalize weight
        weighted_top_n = int(top_n * weight)
        
        # Add items to final_list without duplicates, respecting the weighted_top_n
        count = 0
        for item in recommended_items:
            if item not in seen_items:
                final_list.append(item)
                seen_items.add(item)
                count += 1
            if count >= weighted_top_n:
                break

    return recommendation_results, final_list

if __name__ == "__main__":
    pkl_file_path = './interaction_sparse_matrix.pkl'  # Path to your pickle file
    top_n = 10  # Specify the number of top items to retrieve
    weights = {
        'pop': 1,
        'item_knn': 1,
        'user_knn': 1,
        'bpr': 1,
        # 'neu_mf': 1,
        'simple_x': 1,
        # 'lightgcn': 1,
    }
    separate_recommendations, final_recommendations = run_recommendation_systems(pkl_file_path, weights, top_n)
    print("The recommendation indices from each channel:")
    print(separate_recommendations)
    print("The weighted combined recommendation indices:")
    print(final_recommendations)

    # Save the final list as a pickle file
    with open('final_recommendations.pkl', 'wb') as f:
        pickle.dump(final_recommendations, f)
    
    # Save the seperate list as a pickle file
    sep_recom_list = list(separate_recommendations.values())
    with open('separate_recommendations.pkl', 'wb') as f:
        pickle.dump(sep_recom_list, f)
'''

