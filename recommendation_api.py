import psycopg2
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

def load_data_and_train():
    conn = psycopg2.connect(
        user="postgres",
        password="edenw",
        host="localhost",
        port="5000",  # Change to 5432 if that's your default
        database="myDump"
    )
    clicks_df = pd.read_sql("""
        SELECT user_id, product_id as item_id, COUNT(*) as click_count
        FROM user_clicks
        GROUP BY user_id, product_id
    """, conn)
    orders_df = pd.read_sql("""
        SELECT o.customer_id as user_id, oi.product_id as item_id, COUNT(*) as order_count
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        GROUP BY o.customer_id, oi.product_id
    """, conn)
    ratings_df = pd.read_sql("""
        SELECT user_id, product_id as item_id, rating
        FROM user_reviews
    """, conn)
    # Prepare user and item mappings (include all users/items from all sources)
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
    return model, user_to_idx, item_to_idx, interaction_matrix

def get_default_recommendations(n=5):
    item_scores = np.array(interaction_matrix.sum(axis=0)).flatten()
    top_indices = np.argsort(item_scores)[::-1][:n]
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    recommendations = []
    for idx in top_indices:
        item_id = int(idx_to_item[idx])
        score = float(item_scores[idx])
        recommendations.append({"item_id": item_id, "score": score})
    return recommendations

model, user_to_idx, item_to_idx, interaction_matrix = load_data_and_train()
app = FastAPI()

class RecommendationRequest(BaseModel):
    user_id: int
    n: int = 5

@app.post("/recommend/")
def recommend(request: RecommendationRequest):
    user_id = request.user_id
    n = request.n
    if user_id not in user_to_idx:
        recommendations = get_default_recommendations(n)
        return {
            "user_id": int(user_id),
            "recommendations": recommendations,
            "note": "User not found, showing default recommendations."
        }
    user_idx = user_to_idx[user_id]
    item_indices, scores = model.recommend(user_idx, interaction_matrix[user_idx], N=n)
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    recommendations = []
    for item_idx, score in zip(item_indices, scores):
        item_id = int(idx_to_item[item_idx])
        recommendations.append({"item_id": item_id, "score": float(score)})
    return {"user_id": int(user_id), "recommendations": recommendations}

# @app.get("/users/")
# def list_users():
#     return {"user_ids": list(user_to_idx.keys())}

@app.get("/")
def root():
    return {"message": "Welcome to the Recommendation API! See /docs for usage."}