import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import altair as alt

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

    enc = OneHotEncoder(drop='first', sparse_output=False)
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

    # ──────────────── Component 2: Bar Charts ────────────────
    st.header("3. Bar Charts")

    raw_df = st.session_state['raw_df']
    target = st.session_state['target']
    cat_cols = raw_df.select_dtypes(include=['object', 'category']).columns.tolist()

    if cat_cols:
        cat_var = st.radio("Select a categorical variable", cat_cols)
        avg_df = (
            raw_df
            .dropna(subset=[cat_var, target])
            .groupby(cat_var)[target]
            .mean()
            .reset_index()
        )

        chart1 = (
            alt.Chart(avg_df)
            .mark_bar()
            .encode(
                x=alt.X(cat_var, sort=None, title=cat_var),
                y=alt.Y(target, title=f"Average {target}")
            )
            .properties(width=600, height=400, title=f"Average {target} by {cat_var}")
        )
        st.altair_chart(chart1, use_container_width=True)
    else:
        st.warning("No categorical columns available for Chart 1.")

    corrs = (
        raw_df
        .corr(numeric_only=True)[target]
        .abs()
        .drop(labels=[target], errors='ignore')
        .sort_values(ascending=False)
    )
    corr_df = corrs.reset_index().rename(columns={'index': 'feature', target: 'correlation'})

    if not corr_df.empty:
        chart2 = (
            alt.Chart(corr_df)
            .mark_bar()
            .encode(
                x=alt.X('feature', sort='-y', title='Feature'),
                y=alt.Y('correlation', title=f"|Corr| with {target}")
            )
            .properties(width=600, height=400, title=f"Absolute Correlations with {target}")
        )
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.warning("Not enough numeric columns to compute correlations.")

if __name__ == '__main__':
    main()
