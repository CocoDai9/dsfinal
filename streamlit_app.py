import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("mushrooms.csv")

label_encoders = {}
df_encoded = df.copy()

for column in df.columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# ===== 数据展示 =====
st.write("## Encoded Dataset")
st.dataframe(df_encoded.head())

# ===== 保存文件到本地 =====
df_encoded.to_csv("mushrooms_encoded.csv", index=False)

# ===== 下载按钮（只保留一个）=====
csv = df_encoded.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Encoded CSV",
    data=csv,
    file_name='mushrooms_encoded.csv',
    mime='text/csv',
)

# ===== Mapping（完全保留）=====
st.write("## 🔁 Encoding Mapping (Letter → Number)")

for col in df.columns:
    mapping = dict(zip(
        label_encoders[col].classes_,
        label_encoders[col].transform(label_encoders[col].classes_)
    ))
    
    st.write(f"### {col}")
    st.write(mapping)