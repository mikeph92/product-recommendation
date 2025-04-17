import streamlit as st
import base64
import importlib
import pandas as pd
import gc
from utils.memory_monitor import display_memory_usage, add_memory_cleanup_button


# ===== Sidebar background =====
def sidebar_bg(side_bg_path):
    ext = side_bg_path.split('.')[-1]
    with open(side_bg_path, "rb") as f:
        side_bg = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
            background: url(data:image/{ext};base64,{side_bg});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    # Load data in chunks to manage memory
    chunk_size = 10000  # Adjust based on your memory constraints
    
    # Load products data
    products_chunks = []
    for chunk in pd.read_csv('data/Products_ThoiTrangNam_clean_part1.csv', chunksize=chunk_size):
        products_chunks.append(chunk)
    for chunk in pd.read_csv('data/Products_ThoiTrangNam_clean_part2.csv', chunksize=chunk_size):
        products_chunks.append(chunk)
    products_clean = pd.concat(products_chunks, ignore_index=True)
    
    # Load ratings data
    rating_clean = pd.read_csv("data/Products_ThoiTrangNam_rating_clean.csv", sep='\t')
    
    # Clear memory
    del products_chunks
    gc.collect()
    
    return products_clean, rating_clean

# ===== Giao diện chính =====
def main():
    sidebar_bg("images/bg.png")

    # Ẩn stSidebarNav
    hide_sidebar_style = '''
        <style>
        [data-testid="stSidebarNav"] {
            display:none !important;
        }
        </style>
        '''
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)

    # Load data with caching
    products_clean, rating_clean = load_data()
    
    # Menu: ánh xạ Tên hiển thị → (file, hàm)
    menu = {
        "🏠 Trang chủ": ("home", "home", []),  # không có tham số
        "📊 Giới thiệu chung": ("general_content", "general_content", []),
        "📈 Khám phá dữ liệu": ("data_insight", "data_insight", [products_clean, rating_clean]),
        "🎯 Gợi ý sản phẩm": ("recommendation", "product_recommendation", [products_clean, rating_clean])
    }

    st.sidebar.title("📌 Chức năng")
    selected = st.sidebar.radio("Chọn trang:", list(menu.keys()))
    
    # Add memory monitoring
    display_memory_usage()
    add_memory_cleanup_button()

    # Gọi đúng module và hàm theo lựa chọn
    module_name, function_name, params = menu[selected]
    module = importlib.import_module(f"pages.{module_name}")
    getattr(module, function_name)(*params)

    # Footer nhóm
    st.sidebar.markdown("""
    <div style="margin-top: 200px; background-color: #e0f2f1; padding: 15px; border-radius: 8px;">
        <strong>DL07 – K302 – April 2025</strong><br>
        Hàn Thảo Anh<br>
        Nguyễn Thị Thùy Trang<br>
        👩‍🏫 <strong>GVHD: Cô Khuất Thùy Phương</strong>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()