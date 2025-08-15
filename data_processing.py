import pandas as pd
import random
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix
import pickle
import os

def generate_random_user_id():
    return random.randint(0, 49)

def process_data(file_path, output_path):

    # Step 1: Read data
    data = pd.read_csv(file_path)

    # Step 2: Generate random user-id
    data['user_id'] = [generate_random_user_id() for _ in tqdm(range(len(data)), desc=f"Generating user IDs for {os.path.basename(file_path)}")]
    data['ad_id'] = data.index

    print(f"Load data from {os.path.basename(file_path)}:")
    print(data.head())  

    # Step 3: Mapping
    user_mapping = {user_id: idx for idx, user_id in enumerate(data['user_id'].unique())}
    ad_mapping = {ad_id: idx for idx, ad_id in enumerate(data['ad_id'].unique())}

    data['user_index'] = data['user_id'].map(user_mapping)
    data['ad_index'] = data['ad_id'].map(ad_mapping)

    # Step 4: Construct sparse-matrix
    rows = data['user_index'].values
    cols = data['ad_index'].values
    values = data['conversionAction'].values

    sparse_matrix = coo_matrix((values, (rows, cols)))

    # Sep 5: Put them into pkl files
    with open(output_path, 'wb') as f:
        pickle.dump(sparse_matrix, f)

    print(f"Interaction sparse matrix has been stored at: '{output_path}'.")

if __name__ == "__main__":
    
    base_dir = '/Users/denghaonan/Desktop/AuctionNet-main/0_channel/data_5'
    
    for i in range(48):
        file_name = f"TSI_{i}.0.csv" 
        file_path = os.path.join(base_dir, file_name)
        
        output_file_name = f"interaction_sparse_matrix_{i}.pkl"  
        output_path = os.path.join(base_dir, output_file_name)
        
        print(f"\nLoading file: {file_path}")
        process_data(file_path, output_path)
