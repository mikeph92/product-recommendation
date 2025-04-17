import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
import re

# ========== Hàm tiền xử lý ==========
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

def search_and_recommend(model_dict, product_df, keyword, top_k=10):
    vectorizer = model_dict["tfidf_vectorizer"]
    tfidf_matrix = model_dict["tfidf_matrix"]
    
    # Preprocess + Tokenize query như dữ liệu gốc
    query_processed = preprocess_text(keyword)
    query_tokenized = word_tokenize(query_processed, format="text")

    # Vector hóa từ khóa và tính similarity
    query_vector = vectorizer.transform([query_tokenized])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Lấy top-k sản phẩm có similarity cao nhất
    top_indices = similarity_scores.argsort()[::-1][:top_k]

    result = product_df.iloc[top_indices].copy()
    result["similarity"] = similarity_scores[top_indices]
    result = result[result['similarity'] > 0]

    return result[["product_id", "product_name", "sub_category", "price", "rating", "description", "similarity", "image", "link"]].head(top_k)


def recommend_by_product_id(model_dict, product_df, product_id, top_k=10):
    # Chuyển đổi kiểu dữ liệu của product_id sang số nguyên
    try:
        product_id = int(product_id)
    except ValueError:
        st.error("❌ Mã sản phẩm không hợp lệ. Vui lòng nhập lại mã sản phẩm dạng số nguyên.")
    
    cosine_sim = model_dict["cosine_similarity"]

    if product_id not in cosine_sim.keys():
        st.error("❌ Mã sản phẩm không tồn tại trong dữ liệu.")
    
    sim_scores = list(enumerate(cosine_sim[product_id]))
    result_ids = [x[1][0] for x in sim_scores][:top_k]
    similarity = [x[1][1] for x in sim_scores][:top_k]
    
    # lấy những sản phẩm từ product_df mà product_id chứa trong result_ids
    result = product_df[product_df['product_id'].isin(result_ids)]
    result = result.copy()
    result["similarity"] = similarity

    return result[["product_id", "product_name", "sub_category", "price", "rating", "description", "similarity", "image", "link"]].head(top_k)