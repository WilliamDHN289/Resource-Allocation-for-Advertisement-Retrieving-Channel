import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.sparse

class LightGCN(nn.Module):
    def __init__(self, user_item_matrix, n_layers=3, embedding_dim=64, lr=0.01, n_epochs=100):
        super(LightGCN, self).__init__()
        
        self.num_users, self.num_items = user_item_matrix.shape
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        
        # User and Item Embeddings
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        
        # Adjacency matrix for user-item interactions (converted to sparse tensor)
        self.adj_matrix = self.create_adj_matrix(user_item_matrix)
        
        # n_total = 1100
        n_total = self.num_users + self.num_items
        
        if self.adj_matrix.shape != (n_total, n_total):
            # Create an empty matrix of size (n_users + n_items, n_users + n_items)
            adjusted_matrix = torch.zeros((n_total, n_total))
            
            # Fill in the user-item interactions
            # adjusted_matrix[:100, 100:] = self.adj_matrix.to_dense()  # User-Item interactions
            # adjusted_matrix[100:, :100] = self.adj_matrix.to_dense().T  # Item-User interactions
            adj_dense = self.adj_matrix.to_dense()
            adjusted_matrix[:self.num_users, self.num_users:] = adj_dense 
            adjusted_matrix[self.num_users:, :self.num_users] = adj_dense.T  
            self.adj_matrix = adjusted_matrix.to_sparse()
            
            # Set the modified adjacency matrix as the new adjacency matrix
            # self.adj_matrix = adjusted_matrix
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def create_adj_matrix(self, user_item_matrix):
        # Check the input tenor type
        if not user_item_matrix.is_sparse:
            raise ValueError("Expected a sparse tensor as input.")
        
        # Take the indices and values of the sparse matrix
        indices = user_item_matrix._indices()  # indices of non-zero elements (2, N)
        values = user_item_matrix._values().float()     # value of non-zero elements (N,)

        # Number of rows
        num_nodes = user_item_matrix.size(0)

        # Low sum
        row_sum = torch.zeros(num_nodes, dtype=torch.float, device=values.device)
        row_sum.scatter_add_(0, indices[0], values)  # Sum on each line
        row_sum = row_sum + 1e-6  # Prevent dividing zero
        row_inv_sqrt = 1. / torch.sqrt(row_sum)

        # Normalization
        row_inv_sqrt_values = row_inv_sqrt[indices[0]]  # 获取每行的逆平方根
        normalized_values = values * row_inv_sqrt_values

        # Construct normalized sparse matrix
        adj_matrix_normalized = torch.sparse.FloatTensor(
            indices, normalized_values, user_item_matrix.size()
        )

        return adj_matrix_normalized

    
    '''
    def create_adj_matrix(self, user_item_matrix):
        # Convert user-item matrix to a sparse adjacency matrix
        user_indices, item_indices = user_item_matrix.nonzero()
        values = user_item_matrix[user_indices, item_indices].A1  # Extract non-zero values as a 1D array

        # Create a sparse tensor
        adj_matrix = torch.sparse.FloatTensor(
        torch.LongTensor([user_indices, item_indices]),  # Indices of non-zero elements (2D tensor)
        torch.FloatTensor(values),       # Non-zero values
        torch.Size(user_item_matrix.shape)  # Shape of the full matrix
        )
        
        indices = adj_matrix._indices()  # (2, N) tensor containing row and column indices
        values = adj_matrix._values()    # Values of the sparse matrix
        num_nodes = adj_matrix.size(0)

        # Compute the degree of each node (sum of rows)
        row_sum = torch.zeros(num_nodes)
        row_sum.scatter_add_(0, indices[0], values)  # Sum by row index

        # Avoid division by zero by adding a small epsilon
        row_sum = row_sum + 1e-6
        row_inv_sqrt = 1. / torch.sqrt(row_sum)

        # Normalize the adjacency matrix
        row_inv_sqrt = row_inv_sqrt[indices[0]]  # Apply the inverse square root to the corresponding rows
        values = values * row_inv_sqrt

        # Rebuild the normalized sparse matrix
        adj_matrix_normalized = torch.sparse.FloatTensor(indices, values, adj_matrix.size())

        return adj_matrix_normalized

    '''
 

    def forward(self):
        """
        Forward pass: propagate embeddings through graph layers.
        """
        # Get user and item embeddings
        user_embeddings = self.user_embedding.weight  # Shape: (n_users, latent_dim)
        item_embeddings = self.item_embedding.weight  # Shape: (n_items, latent_dim)
        
        # Concatenate user and item embeddings
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)  # Shape: (n_users + n_items, latent_dim)

        # Propagate embeddings through graph layers
        for _ in range(self.n_layers):
            # Propagate embeddings using adjacency matrix
            all_embeddings = torch.matmul(self.adj_matrix, all_embeddings)  # Shape: (n_users + n_items, latent_dim)
        
        return all_embeddings
 
    '''
    def fit(self):
        """
        Train the model using the adjacency matrix for the specified number of epochs.
        """
        for epoch in range(self.n_epochs):
            self.train()
            self.optimizer.zero_grad()

            all_embeddings = self.forward()

            # Get the user and item embeddings
            user_embeddings = all_embeddings[:self.num_users]
            item_embeddings = all_embeddings[self.num_users:]
            
            # Compute the prediction (dot product)
            prediction = torch.matmul(user_embeddings, item_embeddings.T)
            
            # Target is binary interaction matrix (implicit feedback)
            target = self.adj_matrix[:100, 100:]

            # Binary cross-entropy loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, target)
            loss.backward()
            self.optimizer.step()
            
            print(f'Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss.item()}')
    '''

    def fit(self):
        """
        Train the model using the adjacency matrix for the specified number of epochs.
        """
        # 将稀疏邻接矩阵转换为密集张量
        adj_dense = self.adj_matrix.to_dense()
        
        # 动态获取用户-物品交互的目标矩阵
        target = adj_dense[:self.num_users, self.num_users:]

        for epoch in range(self.n_epochs):
            self.train()
            self.optimizer.zero_grad()

            # 前向传播
            all_embeddings = self.forward()

            # 获取用户和物品的嵌入
            user_embeddings = all_embeddings[:self.num_users]
            item_embeddings = all_embeddings[self.num_users:]
            
            # 计算预测值（点积）
            prediction = torch.matmul(user_embeddings, item_embeddings.T)
            
            # 使用二元交叉熵损失
            loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, target)
            loss.backward()
            self.optimizer.step()
            
            print(f'Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss.item()}')


    def predict(self, user_id):
        """
        Predict the scores for the given user_id for all items.
        """
        all_embeddings = self.forward()
        user_embedding = all_embeddings[user_id]
        scores = torch.matmul(user_embedding, self.item_embedding.weight.T)
        return scores.detach().cpu().numpy()
    
if __name__ == "__main__":
    # 创建一个简单的用户-物品交互矩阵（例如，5个用户和10个物品）
    user_item_matrix = np.array([
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
    ])

    # 将用户-物品交互矩阵转换为稀疏矩阵（torch.sparse要求用CSR格式）
    user_item_matrix_sparse = torch.sparse_coo_tensor(
        indices=np.array(user_item_matrix.nonzero()),
        values=user_item_matrix[user_item_matrix.nonzero()],
        size=user_item_matrix.shape
    )

    # 初始化 LightGCN 模型
    light_gcn = LightGCN(
        user_item_matrix=user_item_matrix_sparse,
        n_layers=3,
        embedding_dim=16,  # 使用较小的嵌入维度以便调试
        lr=0.01,
        n_epochs=5  # 设置较少的训练周期以便快速验证
    )

    # 打印模型结构
    print(light_gcn)

    # 测试模型训练方法
    print("\n开始训练模型...")
    light_gcn.fit()

    # 测试预测功能
    print("\n为用户0生成推荐分数：")
    user_0_scores = light_gcn.predict(user_id=0)
    print(user_0_scores)