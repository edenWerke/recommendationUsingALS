import psycopg2
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

def load_data_and_train():
    # Connect to your PostgreSQL database
    conn = psycopg2.connect(
        user="postgres",
        password="edenw",
        host="localhost",
        port="5000",  # Change to 5432 if that's your default
        database="myDump"
    )
    print("Connected to PostgreSQL!")

    # Load click and order data with time of day
    clicks_df = pd.read_sql("""
        SELECT 
            user_id, 
            product_id as item_id, 
            COUNT(*) as click_count,
            EXTRACT(HOUR FROM clicked_at) as hour_of_day
        FROM user_clicks
        GROUP BY user_id, product_id, EXTRACT(HOUR FROM clicked_at)
    """, conn)
    
    orders_df = pd.read_sql("""
        SELECT 
            o.customer_id as user_id, 
            oi.product_id as item_id, 
            COUNT(*) as order_count,
            EXTRACT(HOUR FROM o.created_at) as hour_of_day
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        GROUP BY o.customer_id, oi.product_id, EXTRACT(HOUR FROM o.created_at)
    """, conn)

    # Load ratings data
    ratings_df = pd.read_sql("""
        SELECT user_id, product_id as item_id, rating
        FROM review
    """, conn)

    # Filter for midnight interactions (hour = 0)
    midnight_clicks = clicks_df[clicks_df['hour_of_day'] == 0]
    midnight_orders = orders_df[orders_df['hour_of_day'] == 0]

    # Prepare user and item mappings (include all users/items from all sources)
    all_users = pd.concat([midnight_clicks['user_id'], midnight_orders['user_id'], ratings_df['user_id']]).unique()
    all_items = pd.concat([midnight_clicks['item_id'], midnight_orders['item_id'], ratings_df['item_id']]).unique()
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}

    # Build interaction matrices for midnight interactions
    click_matrix = sparse.csr_matrix(
        (midnight_clicks['click_count'],
         (midnight_clicks['user_id'].map(user_to_idx), midnight_clicks['item_id'].map(item_to_idx))),
        shape=(len(user_to_idx), len(item_to_idx))
    )
    order_matrix = sparse.csr_matrix(
        (midnight_orders['order_count'],
         (midnight_orders['user_id'].map(user_to_idx), midnight_orders['item_id'].map(item_to_idx))),
        shape=(len(user_to_idx), len(item_to_idx))
    )
    # Ratings matrix (weight: 3)
    rating_matrix = sparse.csr_matrix(
        (ratings_df['rating'],
         (ratings_df['user_id'].map(user_to_idx), ratings_df['item_id'].map(item_to_idx))),
        shape=(len(user_to_idx), len(item_to_idx))
    )
    
    # Combine clicks, orders, and ratings (orders weighted highest, then ratings, then clicks)
    interaction_matrix = click_matrix + (order_matrix * 5) + (rating_matrix * 3)

    # Train ALS model
    model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=15)
    model.fit(interaction_matrix)
    conn.close()
    print("Model trained!")
    return model, user_to_idx, item_to_idx, interaction_matrix

if __name__ == "__main__":
    model, user_to_idx, item_to_idx, interaction_matrix = load_data_and_train()
    idx_to_item = {v: k for k, v in item_to_idx.items()}

    # List all users
    print("Valid user IDs:", list(user_to_idx.keys()))
    # Ask for a user ID
    user_id = int(input("Enter a user ID from above: "))
    n = int(input("How many recommendations? "))

    if user_id not in user_to_idx:
        print("User not found in the data.")
    else:
        user_idx = user_to_idx[user_id]
        item_indices, scores = model.recommend(user_idx, interaction_matrix[user_idx], N=n)
        recommendations = [(int(idx_to_item[item_idx]), float(score)) for item_idx, score in zip(item_indices, scores)]
        print(f"Midnight recommendations for user {user_id}:")
        for item_id, score in recommendations:
            print(f"  Item {item_id} (score: {score:.2f})")