import os
import warnings
import logging
import gc
# Ẩn tất cả warnings 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold(absl.logging.ERROR)
import streamlit as st
import joblib
import os
from tensorflow.keras.models import load_model
from utils.content_based_top20 import *
from utils.collaborative import get_top_n_recommendations

# Configure TensorFlow to use memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

# ====== Load mô hình & dữ liệu ======
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_cb_model():
    path = "models/content_based_model.pkl"
    if not os.path.exists(path):
        st.error("❌ Không tìm thấy file content_based_model.pkl")
        st.stop()
    model = joblib.load(path)
    # Optimize memory usage
    if 'tfidf_matrix' in model:
        model['tfidf_matrix'] = model['tfidf_matrix'].tocsr()  # Convert to CSR format for better memory efficiency
    return model

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_cf_model():
    path_model = "models/matrix_factorizer_keras_model.h5"
    path_meta = "models/id_mappings.pkl"

    if not os.path.exists(path_model):
        st.error("❌ Không tìm thấy file models/matrix_factorizer_keras_model.h5")
        st.stop()
    elif not os.path.exists(path_meta):
        st.error("❌ Không tìm thấy file models/id_mappings.pkl")
        st.stop()
    
    # Load model with memory optimization
    model = load_model(path_model, compile=False)
    meta = joblib.load(path_meta)
    
    # Clear any unused memory
    gc.collect()
    
    return model, meta


# ====== Hiển thị sản phẩm gợi ý ======
def display_recommendations(result_df, is_cb=True):
    if result_df is None or result_df.empty:
        st.warning("🙁 Không tìm thấy sản phẩm phù hợp.")
    else:
        for _, row in result_df.iterrows():
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    try:
                        st.image(row['image'], width=120)
                    except:
                        st.image("images/no_image.jpg", width=120)

                with cols[1]:
                    mota = str(row['description'])
                    short_desc = mota[:200] + "..." if len(mota) > 200 else mota

                    st.markdown(f"""
                    **🧢 Tên sản phẩm:** {row['product_name']}  
                    **📦 Loại sản phẩm:** {row['sub_category']}  
                    **💸 Giá:** {int(row['price']):,}₫  
                    **⭐ Đánh giá:** {float(row['rating']):.1f}  
                    **📖 Mô tả:** {short_desc}
                    """)

                    if is_cb and 'similarity' in row:
                        st.markdown(f"📊 **Độ tương đồng:** {float(row['similarity']):.3f}")
                    elif not is_cb and 'predict' in row and float(row['predict']) > 0:
                        st.markdown(f"📊 **Dự đoán:** {float(row['predict']):.1f}")
                        
                    st.markdown(f"""
                                <a href="{row['link']}" target="_blank">👉 Xem chi tiết sản phẩm</a>
                                """, unsafe_allow_html=True)
                st.markdown("---")

def display_carousel(result_df, num_cols=3):
    if result_df is None or result_df.empty:
        st.warning("🙁 Khách hàng chưa có lịch sử đánh giá sản phẩm.")
        return

    # Chia dữ liệu thành từng "hàng" (chunk), mỗi hàng có tối đa num_cols sản phẩm
    rows = [result_df[i:i+num_cols] for i in range(0, len(result_df), num_cols)]

    for row_chunk in rows:
        cols = st.columns(num_cols)

        for i in range(num_cols):
            with cols[i]:
                if i < len(row_chunk):
                    row = row_chunk.iloc[i]

                    st.markdown("---")
                    st.markdown(
                        f"""
                        <div style='text-align: right; margin-bottom: 0.5rem;'>
                            <a href="{row['link']}" target="_blank">🔗 Xem chi tiết</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    try:
                        st.image(row['image'], use_container_width=True)
                    except:
                        st.image("images/no_image.jpg", use_container_width=True)

                    st.markdown(f"**{row['product_name']}**")
                    st.markdown(f"💸 **{int(row['price']):,}₫**")
                    st.markdown(f"⭐ {float(row['rating']):.1f}")

                    short_desc = str(row['description'])[:100] + "..." if row['description'] else "Không có mô tả"
                    st.caption(short_desc)
                else:
                    st.write("")  # Cột trống nếu không còn sản phẩm

# ====== Giao diện chính gợi ý ======
def product_recommendation(products_df, ratings_df):
    st.header("🎯 Hệ thống gợi ý sản phẩm")

    method = st.selectbox("🔍 Chọn phương pháp gợi ý:", ["Gợi ý theo nội dung", "Gợi ý theo người dùng"])
    
    # Load models only when needed
    if method == "Gợi ý theo nội dung":
        model_cb = load_cb_model()
    else:
        model_cf, meta = load_cf_model()
        user_mapping = meta["user_mapping"]
        product_mapping = meta["product_mapping"]
        mu = meta["mu"]

    if method == "Gợi ý theo nội dung":
        search_mode = st.radio("Chọn cách tìm kiếm:", ["Từ khóa", "Mã sản phẩm"])

        if search_mode == "Từ khóa":
            keyword = st.text_input("Nhập từ khóa (ví dụ: áo thun)")
            if st.button("Gợi ý", key="btn_cb_keyword"):
                result = search_and_recommend(model_cb, products_df, keyword, top_k=10)
                display_recommendations(result, is_cb=True)

        elif search_mode == "Mã sản phẩm":
            my_dict = list(model_cb["cosine_similarity"].items())
            unique_ids = [x[0] for x in my_dict]

            product_id = st.selectbox("Chọn mã sản phẩm:", unique_ids)

            if st.button("Gợi ý", key="btn_cb_product"):
                try:
                    result = recommend_by_product_id(model_cb, products_df, product_id, top_k=10)
                    product = products_df[products_df['product_id'] == product_id]
                    
                    st.markdown(f"### 🛒 <span style='color:#1f77b4'>Sản phẩm đang xem:</span> <strong>{product_id}</strong>", unsafe_allow_html=True)
                    display_recommendations(product, is_cb=True)
                    
                    st.markdown("### 🎯 <span style='color:#2ca02c'>Sản phẩm gợi ý:</span>", unsafe_allow_html=True)
                    display_recommendations(result, is_cb=True)
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    elif method == "Gợi ý theo người dùng":
        user_map_df = ratings_df[['user_id', 'user']].drop_duplicates()
        user_map = dict(zip(user_map_df['user_id'], user_map_df['user']))

        st.subheader("👥 Danh sách người dùng và mã ID")
        st.dataframe(user_map_df.reset_index(drop=True), use_container_width=True)
        
        user_ids = user_map.keys()
        st.markdown("""
        ### 🔑 <span style='color:#0e76a8;'>Nhập mã khách hàng</span>  
        <small><i>Ví dụ: <code>5</code>, <code>290</code>, <code>777</code>, <code>20000</code></i></small>
        """, unsafe_allow_html=True)

        selected_user = st.text_input(" ", key="user_input")

        if st.button("Gợi ý", key="btn_cf_user"):
            if selected_user:
                try:
                    user_id = int(selected_user)  # nếu là số
                    if user_id not in user_ids:
                        st.error("⚠️ Mã khách hàng không tồn tại trong hệ thống hoặc chưa có lịch sử đánh giá!")
                    else:
                        user_name = user_map.get(user_id, "Không xác định")

                        st.markdown(f"""
                        <div style="font-size:20px; font-weight:600;">
                            👤 <span style="color:#0e76a8;">Tên người dùng:</span> {user_name}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("####")
                        st.subheader("🛍️ Sản phẩm đã đánh giá:")
                        user_rated_df = ratings_df[ratings_df['user_id'] == user_id]
                        rated_products = products_df[products_df['product_id'].isin(user_rated_df['product_id'])].copy()

                        display_carousel(rated_products)

                        result = get_top_n_recommendations(
                            product_df=products_df,
                            user_id=user_id,
                            model=model_cf,
                            user_mapping=user_mapping,
                            product_mapping=product_mapping,
                            mu=mu,
                            rated_products=rated_products,
                            n=10
                        )
                                                        
                        st.markdown("##")
                        st.subheader("🎁 Gợi ý sản phẩm dựa trên hành vi người dùng:")
                        display_recommendations(result, is_cb=False)
                except ValueError:
                    st.error("❌ Mã khách hàng phải là một số nguyên!")
                except Exception as e:
                    st.error(f"Lỗi khi gợi ý: {e}")
            else:
                st.error("⚠️ Vui lòng nhập Mã khách hàng!")