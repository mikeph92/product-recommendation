import pandas as pd

# Đọc file CSV gốc vào DataFrame
df = pd.read_csv('data/Products_ThoiTrangNam_clean.csv')

# Chia DataFrame thành 2 phần bằng nhau
half_size = len(df) // 2  # Xác định điểm chia
df_part1 = df.iloc[:half_size]  # Phần 1
df_part2 = df.iloc[half_size:]  # Phần 2

# Lưu 2 phần vào 2 file CSV mới
df_part1.to_csv('data/Products_ThoiTrangNam_clean_part1.csv', index=False)
df_part2.to_csv('data/Products_ThoiTrangNam_clean_part2.csv', index=False)

# Đọc lại 2 file thành DataFrame
df1 = pd.read_csv('data/Products_ThoiTrangNam_clean_part1.csv')
df2 = pd.read_csv('data/Products_ThoiTrangNam_clean_part2.csv')

# Gộp lại thành một DataFrame duy nhất
df_combined = pd.concat([df1, df2], ignore_index=True)

# Hiển thị kết quả
print(df_combined)
