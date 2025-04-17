import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm 
import numpy as np
from underthesea import word_tokenize

# ========== Cài đặt ==========
input_file = "data/Products_ThoiTrangNam_clean.csv"
output_pkl = "models/content_based_model.pkl"

# ========== Hàm tiền xử lý ==========
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

# ========== Đọc & lọc dữ liệu ==========
print("📦 Đang load dữ liệu...")
df = pd.read_csv(input_file)

# Xử lý mô tả
df["combined_text"] = (df["product_name"].fillna("") + " " + df["clean_description"].fillna("")).apply(preprocess_text)

# word_tokenize
df["combined_text_wt"]=df["combined_text"].apply(lambda x: word_tokenize(x, format="text"))

# Đọc stopwords từ file
STOP_WORD_FILE = 'data/vietnamese-stopwords.txt'
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()

stop_words = stop_words.split('\n')

# ========== Xây dựng mô hình TF-IDF ==========
print("🔍 Đang vector hóa TF-IDF...")
vectorizer = TfidfVectorizer(analyzer='word', stop_words=stop_words)
tfidf_matrix = vectorizer.fit_transform(df["combined_text_wt"])

# ========== Tính toán cosine similarity ==========
item_ids = df['product_id'].tolist()
top_k = 20
top_k_dict = {}

print("🔄 Đang tính cosine similarity từng hàng")

# Lặp qua từng hàng
for idx in tqdm(range(tfidf_matrix.shape[0])):
    # Lấy vector tf-idf của sản phẩm hiện tại
    item_vector = tfidf_matrix[idx]

    # Tính cosine similarity với toàn bộ tf-idf matrix
    similarity_row = cosine_similarity(item_vector, tfidf_matrix).flatten()

    # Loại bỏ chính nó
    similarity_row[idx] = -1

    # Lấy top-k
    top_indices = np.argsort(similarity_row)[-top_k:][::-1]
    top_items = [(item_ids[i], float(similarity_row[i])) for i in top_indices]

    top_k_dict[item_ids[idx]] = top_items

# ========== Lưu mô hình ==========
print("💾 Đang lưu mô hình .pkl ...")
model = {
    "tfidf_vectorizer": vectorizer,
    "tfidf_matrix": tfidf_matrix,
    "cosine_similarity": top_k_dict
}

joblib.dump(model, output_pkl)
print(f"🎉 Mô hình đã lưu vào {output_pkl}")