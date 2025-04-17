import streamlit as st

def home():
    st.image("images/home.png", width=1000)
    
    # Hiển thị tiêu đề đồ án
    st.title("Đồ án tốt nghiệp Data Science And Machine Learning")
    
    # Hiển thị thông tin chi tiết
    st.subheader("Topic 2: Recommender System")
    st.markdown("**Ngày báo cáo:** 20/4/2025")
    
    # Thêm thông tin người thực hiện
    st.markdown("**Người thực hiện:** Hàn Thảo Anh, Nguyễn Thị Thùy Trang")
    
    # Thêm thông tin về giáo viên hướng dẫn
    st.markdown("**Giáo viên hướng dẫn:** Cô Khuất Thùy Phương")
