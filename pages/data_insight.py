import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objs as go


# Trang phân cụm khách hàng
def data_insight(products_clean, rating_clean):
    st.image("images/insight.jpeg", width=1000)
    st.title("Một số thông tin về dữ liệu")

    st.markdown("### 🛍️ Dữ liệu sản phẩm")
    st.dataframe(products_clean.head(10))
    st.markdown("### ⭐ Dữ liệu đánh giá sản phẩm")
    st.dataframe(rating_clean.head(10))

    
    # Tính các chỉ số
    num_products = products_clean['product_id'].nunique()
    num_users = rating_clean['user_id'].nunique()
    num_ratings = rating_clean.shape[0]
    
    # Merge để lấy thông tin price, product_name
    eda_merged = rating_clean.merge(products_clean[['product_id', 'price', 'product_name']], on='product_id', how='left')

    # User đánh giá nhiều nhất
    top_reviewer = rating_clean['user_id'].value_counts().idxmax()
    top_reviewer_count = rating_clean['user_id'].value_counts().max()

    # User chi tiêu nhiều nhất
    user_spend = eda_merged.groupby('user_id')['price'].sum()
    top_spender = user_spend.idxmax()
    top_spend_amount = user_spend.max()

    # User rating 5 sao nhiều nhất
    one_star_users = rating_clean[rating_clean['rating'] == 5]['user_id'].value_counts()
    top_one_star_user = one_star_users.idxmax()
    top_one_star_count = one_star_users.max()

    # Sản phẩm bán chạy nhất (nhiều đánh giá nhất)
    top_product_id = rating_clean['product_id'].value_counts().idxmax()
    top_product_name = products_clean.loc[products_clean['product_id'] == top_product_id, 'product_name'].values[0]

    # Sản phẩm bán ít nhất (ít đánh giá nhất)
    least_product_id = rating_clean['product_id'].value_counts().idxmin()
    least_product_name = products_clean.loc[products_clean['product_id'] == least_product_id, 'product_name'].values[0]

    col1, col2, col3 = st.columns(3)

    # ---------- CỘT 1: THỐNG KÊ CHUNG ----------
    with col1:
        st.markdown("### 📊 Thống kê tổng quan")
        st.markdown(f"🛍️ **Số sản phẩm:**<br><span style='font-size:16px'>{num_products}</span>", unsafe_allow_html=True)
        st.markdown(f"👤 **Số người dùng:**<br><span style='font-size:16px'>{num_users}</span>", unsafe_allow_html=True)
        st.markdown(f"⭐ **Số lượt đánh giá:**<br><span style='font-size:16px'>{num_ratings}</span>", unsafe_allow_html=True)

    # ---------- CỘT 2: USER ----------
    with col2:
        st.markdown("### 🙋‍♂️ Người dùng nổi bật")
        st.markdown(f"✍️ **Review nhiều nhất:**<br><span style='font-size:15px'>{top_reviewer} ({top_reviewer_count} lần)</span>", unsafe_allow_html=True)
        st.markdown(f"💰 **Chi tiêu nhiều nhất:**<br><span style='font-size:15px'>{top_spender} ({top_spend_amount:,.0f} VNĐ)</span>", unsafe_allow_html=True)
        st.markdown(f"😍 **Rating 5⭐ nhiều nhất:**<br><span style='font-size:15px'>{top_one_star_user} ({top_one_star_count} lần)</span>", unsafe_allow_html=True)

    # ---------- CỘT 3: SẢN PHẨM ----------
    with col3:
        st.markdown("### 📦 Sản phẩm nổi bật")
        st.markdown(f"🔥 **Nhiều đánh giá nhất:**<br><span style='font-size:15px'>{top_product_name[:30]}...</span>", unsafe_allow_html=True)
        st.markdown(f"🥶 **Ít đánh giá nhất:**<br><span style='font-size:15px'>{least_product_name[:30]}...</span>", unsafe_allow_html=True)


    # Tính top nhóm hàng phổ biến
    top_subcat = products_clean['sub_category'].value_counts().head(10)
    # Tiêu đề
    st.subheader("📦 Top 10 Nhóm Hàng Phổ Biến Nhất")

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=top_subcat.values,
        y=top_subcat.index,
        palette="Blues_d",
        ax=ax
    )
    ax.set_title("Top 10 nhóm hàng phổ biến")
    ax.set_xlabel("Số lượng sản phẩm")
    ax.set_ylabel("Nhóm hàng")
    plt.tight_layout()

    # Hiển thị trong Streamlit
    st.pyplot(fig)
    
    # Tính top 20 tên sản phẩm lặp lại nhiều nhất
    top_names = products_clean['product_name'].value_counts().head(20)

    # Hiển thị tiêu đề
    st.subheader("🛍️ Top 20 Tên Sản Phẩm Phổ Biến Nhất")

    # Vẽ biểu đồ với màu
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=top_names.values,
        y=top_names.index,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Top 20 tên sản phẩm phổ biến nhất")
    ax.set_xlabel("Số lần xuất hiện")
    ax.set_ylabel("Tên sản phẩm")
    plt.tight_layout()

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)
    
    st.subheader("🎻 Phân bố giá theo nhóm sản phẩm")

    # Vẽ biểu đồ violin
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=products_clean,
        x='sub_category',
        y='price',
        inner='quartile',
        palette='pastel',
        ax=ax
    )

    # Tùy chỉnh trục
    ax.set_title("Phân bố giá theo nhóm sản phẩm (Violin Plot)")
    ax.set_xlabel("Nhóm sản phẩm")
    ax.set_ylabel("Giá (VNĐ)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Hiển thị trong Streamlit
    st.pyplot(fig)
    
    st.subheader("📦 Phân bố Rating theo Nhóm Sản Phẩm")

    # Vẽ biểu đồ boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=products_clean,
        x='sub_category',
        y='rating',
        palette='Set3',
        ax=ax
    )

    # Tùy chỉnh trục
    ax.set_title("Phân bố rating theo từng nhóm sản phẩm")
    ax.set_xlabel("Nhóm sản phẩm")
    ax.set_ylabel("Rating")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Hiển thị trong Streamlit
    st.pyplot(fig)
    
    st.subheader("📈 Tương Quan giữa Giá và Rating theo Nhóm Sản Phẩm")

    # Vẽ biểu đồ scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=products_clean,
        x='rating',
        y='price',
        hue='sub_category',
        alpha=0.7,
        palette='tab10',
        ax=ax
    )

    # Tùy chỉnh biểu đồ
    ax.set_title("Tương quan giữa giá và rating theo nhóm sản phẩm")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Giá sản phẩm (VNĐ)")
    ax.legend(title='Nhóm sản phẩm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)
    
    # Tính giá trung bình theo nhóm sản phẩm
    avg_price = products_clean.groupby('sub_category')['price'].mean().sort_values()

    # Tiêu đề
    st.subheader("💰 Giá Trung Bình Theo Nhóm Sản Phẩm")

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=avg_price.values,
        y=avg_price.index,
        palette='viridis',
        ax=ax
    )
    ax.set_title("Giá trung bình theo nhóm sản phẩm")
    ax.set_xlabel("Giá trung bình (VNĐ)")
    ax.set_ylabel("Nhóm sản phẩm")
    plt.tight_layout()

    # Hiển thị trong Streamlit
    st.pyplot(fig)
    
    # Tính rating trung bình theo nhóm sản phẩm
    avg_rating = products_clean.groupby('sub_category')['rating'].mean().sort_values()

    # Tiêu đề
    st.subheader("⭐ Rating Trung Bình Theo Nhóm Sản Phẩm")

    # Vẽ biểu đồ barplot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=avg_rating.values,
        y=avg_rating.index,
        palette='magma',
        ax=ax
    )

    # Tùy chỉnh tiêu đề và nhãn
    ax.set_title("Rating trung bình theo nhóm sản phẩm")
    ax.set_xlabel("Rating trung bình")
    ax.set_ylabel("Nhóm sản phẩm")
    plt.tight_layout()

    # Hiển thị trong Streamlit
    st.pyplot(fig)

        
    # Tính toán tỷ lệ đánh giá
    values = rating_clean.rating.value_counts()
    labels = values.index
    colors = ['red', 'blue', 'green', 'yellow', 'black']

    # Tạo pie chart
    trace = go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors)
    )

    # Tạo layout cho pie chart
    layout = go.Layout(
        title='Biểu đồ Ratings theo phần trăm'
    )

    # Tạo figure và vẽ biểu đồ
    fig = go.Figure(data=trace, layout=layout)

    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig)
    
    # Lọc các sản phẩm có rating = 5
    five_star_ratings = rating_clean[rating_clean['rating'] == 5]
    five_star_ratings = five_star_ratings.merge(products_clean[['product_id', 'product_name', 'sub_category']], on='product_id', how='left')

    # Nhóm sản phẩm và đếm số lần đánh giá 5 sao
    top_product = five_star_ratings['product_name'].value_counts().reset_index()
    top_product.columns = ['product_name', 'count']

    # Sắp xếp sản phẩm theo số lượng rating 5 sao giảm dần
    top_product = top_product.sort_values(by='count', ascending=False)

    # Chọn top N sản phẩm có số lượng rating 5 sao nhiều nhất (ví dụ top 10)
    top_N_product = top_product.head(10)

    # Tiêu đề cho phần trong Streamlit
    st.subheader("⭐ Top Sản Phẩm Nhận Được Nhiều Đánh Giá 5 Sao")

    # Vẽ biểu đồ barplot
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(
        data=top_N_product,
        x='count',
        y='product_name',
        palette='viridis',
        ax=ax
    )

    # Tùy chỉnh biểu đồ
    ax.set_xlabel('Count of 5-Star Ratings')
    ax.set_ylabel('Product Name')
    ax.set_title('Product with the Most 5-Star Ratings')

    # Hiển thị biểu đồ trong Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    
    # Nhóm theo sub_category và đếm số lần đánh giá 5 sao
    top_sub_category = five_star_ratings['sub_category'].value_counts().reset_index()
    top_sub_category.columns = ['sub_category', 'count']

    # Sắp xếp các nhóm sản phẩm theo số lượng rating 5 sao giảm dần
    top_sub_category = top_sub_category.sort_values(by='count', ascending=False)

    # Chọn top N nhóm sản phẩm có số lượng rating 5 sao nhiều nhất (ví dụ top 10)
    top_N_sub_category = top_sub_category.head(10)

    # Tiêu đề cho phần trong Streamlit
    st.subheader("⭐ Top Nhóm Sản Phẩm Nhận Được Nhiều Đánh Giá 5 Sao")

    # Vẽ biểu đồ barplot
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(
        data=top_N_sub_category,
        x='count',
        y='sub_category',
        palette='viridis',
        ax=ax
    )

    # Tùy chỉnh biểu đồ
    ax.set_xlabel('Count of 5-Star Ratings')
    ax.set_ylabel('Sub Category')
    ax.set_title('Sub Category with the Most 5-Star Ratings')

    # Hiển thị biểu đồ trong Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    
    # Lọc các sản phẩm có rating = 1
    one_star_ratings = rating_clean[rating_clean['rating'] == 1]

    # Kết hợp dữ liệu từ eda_df và products_clean để có được product_name
    one_star_ratings = one_star_ratings.merge(products_clean[['product_id', 'product_name']], on='product_id', how='left')

    # Nhóm sản phẩm và đếm số lần đánh giá 1 sao
    top_product = one_star_ratings['product_name'].value_counts().reset_index()
    top_product.columns = ['product_name', 'count']

    # Sắp xếp sản phẩm theo số lượng rating 1 sao giảm dần
    top_product = top_product.sort_values(by='count', ascending=False)

    # Chọn top N sản phẩm có số lượng rating 1 sao nhiều nhất (ví dụ top 10)
    top_N_product = top_product.head(10)

    # Tiêu đề cho phần trong Streamlit
    st.subheader("⚠️ Top Sản Phẩm Nhận Được Nhiều Đánh Giá 1 Sao")

    # Vẽ biểu đồ barplot
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(
        data=top_N_product,
        x='count',
        y='product_name',
        palette='viridis',
        ax=ax
    )

    # Tùy chỉnh biểu đồ
    ax.set_xlabel('Count of 1-Star Ratings')
    ax.set_ylabel('Product Name')
    ax.set_title('Product with the Most 1-Star Ratings')

    # Hiển thị biểu đồ trong Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("📊 Phân bố độ dài mô tả sản phẩm")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(products_clean['desc_len'], bins=50, kde=True, ax=ax)
    ax.set_title("Phân bố độ dài mô tả sản phẩm")
    ax.set_xlabel("Số từ")
    ax.set_ylabel("Số sản phẩm")
    plt.tight_layout()

    st.pyplot(fig)
    
    st.markdown("### ☁️ Wordcloud mô tả sản phẩm")
    st.image("images/wordcloud.png", width=1000)