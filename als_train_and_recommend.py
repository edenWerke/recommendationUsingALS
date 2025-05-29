import psycopg2
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

def load_data_and_train():
    conn = psycopg2.connect(
        user="postgres",
        password="edenw",
        host="localhost",
        port="5000",
        database="myDump"
    )
    print("Connected to PostgreSQL!")

    clicks_df = pd.read_sql("""
        SELECT 
            user_id, 
            product_id as item_id, 
            COUNT(*) as click_count
        FROM user_clicks
        GROUP BY user_id, product_id
    """, conn)
    
    orders_df = pd.read_sql("""
        SELECT 
            o.customer_id as user_id, 
            oi.product_id as item_id, 
            COUNT(*) as order_count
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        GROUP BY o.customer_id, oi.product_id
    """, conn)

    ratings_df = pd.read_sql("""
        SELECT user_id, product_id as item_id, rating
        FROM review
    """, conn)

    all_users = pd.concat([clicks_df['user_id'], orders_df['user_id'], ratings_df['user_id']]).unique()
    all_items = pd.concat([clicks_df['item_id'], orders_df['item_id'], ratings_df['item_id']]).unique()
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}

    click_matrix = sparse.csr_matrix(
        (clicks_df['click_count'],
         (clicks_df['user_id'].map(user_to_idx), clicks_df['item_id'].map(item_to_idx))),
        shape=(len(user_to_idx), len(item_to_idx))
    )
    order_matrix = sparse.csr_matrix(
        (orders_df['order_count'],
         (orders_df['user_id'].map(user_to_idx), orders_df['item_id'].map(item_to_idx))),
        shape=(len(user_to_idx), len(item_to_idx))
    )
    rating_matrix = sparse.csr_matrix(
        (ratings_df['rating'],
         (ratings_df['user_id'].map(user_to_idx), ratings_df['item_id'].map(item_to_idx))),
        shape=(len(user_to_idx), len(item_to_idx))
    )
    
    interaction_matrix = click_matrix + (order_matrix * 5) + (rating_matrix * 3)

    model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=15)
    model.fit(interaction_matrix)
    conn.close()
    print("Model trained!")
    return model, user_to_idx, item_to_idx, interaction_matrix

if __name__ == "__main__":
    model, user_to_idx, item_to_idx, interaction_matrix = load_data_and_train()
    idx_to_item = {v: k for k, v in item_to_idx.items()}

    print("Valid user IDs:", list(user_to_idx.keys()))
    user_id = int(input("Enter a user ID from above: "))
    n = int(input("How many recommendations? "))

    if user_id not in user_to_idx:
        print("User not found in the data.")
    else:
        user_idx = user_to_idx[user_id]
        item_indices, scores = model.recommend(user_idx, interaction_matrix[user_idx], N=n)
        recommendations = [(int(idx_to_item[item_idx]), float(score)) for item_idx, score in zip(item_indices, scores)]
        print(f"Recommendations for user {user_id}:")
        for item_id, score in recommendations:
            print(f"  Item {item_id} (score: {score:.2f})")