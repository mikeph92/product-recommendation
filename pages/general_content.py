import streamlit as st

def general_content():
    st.image("images/recommender.png", width=1200) 
    st.title("Chào mừng đến với dự án tạo hệ thống gợi ý sản phẩm cho Shopee")
    
    # Nội dung Business Objective
    st.subheader("🎯 Mục tiêu kinh doanh")
    st.write("""
    Hệ sinh thái Shopee, trong đó có shopee.vn, là một website thương mại điện tử “all in one” hàng đầu của Việt Nam và khu vực Đông Nam Á. 

    
    Yêu cầu: Triển khai hệ thống Recommender System cho website shopee.vn hỗ trợ nâng cao trải nghiệm người dùng và xây dựng nhiều tiện ích khác.
    """)
    
    st.image("images/shopee.png", width=1200) 
    
    # Nội dung dữ liệu
    st.subheader("📂 Dữ liệu")
    st.write("""
    Dữ liệu được thu thập từ Shopee.vn, bao gồm hai tập dữ liệu chính: Products_ThoiTrangNam_raw.csv và 
    Products_ThoiTrangNam_rating_raw.csv""")
    
    # Nội dung về cách giải quyết mục tiêu kinh doanh
    st.subheader("🚀 Quy trình triển khai hệ thống gợi ý")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 1️⃣ Tiền xử lý dữ liệu")
        st.markdown("- Làm sạch dữ liệu, xử lý dữ liệu trùng lặp, thiếu, không hợp lệ, xóa ký tự không cần thiết\n- Chuẩn hóa, tạo các cột mới\n- Xử lý văn bản (tokenize, remove stopwords, vector hóa)")

    with col2:
        st.markdown("### 2️⃣ Phân tích dữ liệu")
        st.markdown("- Thống kê tổng quan\n- Trực quan hóa\n- Tìm hiểu hành vi và xu hướng mua sắm")

    with col3:
        st.markdown("### 3️⃣ Gợi ý sản phẩm")
        st.markdown("- Content-based (Gensim, Cosine)\n- Collaborative (Surprise, ALS, Keras Matrix Factorization)")

    with col4:
        st.markdown("### 4️⃣ Đánh giá mô hình")
        st.markdown("- RMSE, MSE, Jaccard Similarity trung bình, tỷ lệ match, thời gian thực thi\n- So sánh mô hình\n- Chọn mô hình phù hợp")
    st.image("images/ppt.png", width=1200) 

    # Nội dung về dữ liệu
    st.subheader("🧾Kết quả thưc hiện")
    st.write("""
    Tùy thuộc vào **độ lớn và nội dung dữ liệu**, việc lựa chọn giữa **Content-based Filtering** và **Collaborative Filtering** cần được cân nhắc kỹ lưỡng.

    📌 Trong trường hợp dữ liệu của bài toán này:

    - ✅ **Cosine Similarity** (Content-based Filtering)  
    - ✅ **Matrix Factorization using Keras** (Collaborative Filtering)  

    👉 là hai phương pháp cho kết quả tốt và ổn định hơn so với các phương pháp khác.

    Việc áp dụng mô hình phù hợp sẽ giúp cải thiện đáng kể **chất lượng gợi ý sản phẩm** và **trải nghiệm người dùng**.
    """)