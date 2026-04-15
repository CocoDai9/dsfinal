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

st.write("Encoded Dataset:")
st.dataframe(df_encoded.head())

st.write("Encoding Mapping:")

for col in df.columns:
    mapping = dict(zip(
        label_encoders[col].classes_,
        label_encoders[col].transform(label_encoders[col].classes_)
    ))
    
    st.write(f"### {col}")
    st.write(mapping)