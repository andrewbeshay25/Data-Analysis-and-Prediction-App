import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@st.cache_data
def load_data(csv_file):
    return pd.read_csv(csv_file)

@st.cache_data
def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    cat_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    enc = OneHotEncoder(drop='first', sparse=False)
    to_encode = [c for c in ['season', 'weathersit'] if c in df.columns]
    if to_encode:
        enc_df = pd.DataFrame(
            enc.fit_transform(df[to_encode]),
            columns=enc.get_feature_names_out(to_encode),
            index=df.index
        )
        df = pd.concat([df.drop(columns=to_encode), enc_df], axis=1)

    scaler = StandardScaler()
    num_cols = [c for c in ['temp', 'windspeed'] if c in df.columns]
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

def main():
    st.title("Data Analysis & Prediction App")
    st.header("1. Upload Your CSV Dataset")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if not uploaded_file:
        st.info("Please upload a CSV file to proceed.")
        return

    raw_df = load_data(uploaded_file)
    st.success(f"Loaded dataset with {raw_df.shape[0]} rows and {raw_df.shape[1]} columns.")

    if st.checkbox("Show raw data preview"):
        st.dataframe(raw_df.head())

    numeric_cols = raw_df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for target selection.")
        return

    st.header("2. Select Target Variable")
    target = st.selectbox("Target variable (numeric)", numeric_cols)
    st.write(f"**{target}** selected as target.")

    processed_df = preprocess_data(raw_df)

    st.session_state['raw_df'] = raw_df
    st.session_state['processed_df'] = processed_df
    st.session_state['target'] = target

    st.success("Data preprocessing complete! Component 1 is ready for downstream steps.")

if __name__ == '__main__':
    main()
