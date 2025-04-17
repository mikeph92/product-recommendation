import os
import joblib
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE


# Đọc dữ liệu
rating_path = "data/Products_ThoiTrangNam_rating_clean.csv"
df = pd.read_csv(rating_path, sep="\t")
print(f"📊 Tổng số dòng dữ liệu: {len(df)}")

# Đánh lại index từ 0 liên tục
df['user_idx'], _ = pd.factorize(df['user_id'])
df['product_idx'], _ = pd.factorize(df['product_id'])

# Mapping gốc ID -> index
user_mapping = dict(zip(df['user_id'], df['user_idx']))
product_mapping = dict(zip(df['product_id'], df['product_idx']))

N = df['user_idx'].nunique()
M = df['product_idx'].nunique()

# Shuffle & chia train/test
df1 = shuffle(df, random_state=42)
cutoff = int(0.8 * len(df))
df_train = df1.iloc[:cutoff]
df_test = df1.iloc[cutoff:]

# Các tham số
K = 10  # số chiều ẩn
mu = df_train.rating.mean()
epochs = 15
reg = 0.0

# Keras model
u = Input(shape=(1,))
m = Input(shape=(1,))

u_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(u)
m_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(m)
u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u)
m_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(m)

x = Dot(axes=2)([u_embedding, m_embedding])
x = Add()([x, u_bias, m_bias])
x = Flatten()(x)

model = Model(inputs=[u, m], outputs=x)
model.compile(
    loss=MeanSquaredError(),
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    metrics=[MSE()]
)

# Train model
r = model.fit(
    x=[df_train.user_idx.values, df_train.product_idx.values],
    y=df_train.rating.values - mu,
    epochs=epochs,
    batch_size=128,
    validation_data=(
        [df_test.user_idx.values, df_test.product_idx.values],
        df_test.rating.values - mu
    )
)


# Lưu model và mapping
os.makedirs("models", exist_ok=True)
model_path = "models/matrix_factorizer_keras_model.h5"
model.save(model_path)

joblib.dump({
    "user_mapping": user_mapping,
    "product_mapping": product_mapping,
    "mu": mu,
}, "models/id_mappings.pkl")

print(f"💾 Mô hình đã lưu tại: {model_path}")