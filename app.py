import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

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

    # 2: Target Variable
    st.header("2. Select Target Variable")
    target = st.selectbox("Target variable (numeric)", numeric_cols)
    st.write(f"**{target}** selected as target.")

    processed_df = preprocess_data(raw_df)

    st.session_state['raw_df'] = raw_df
    st.session_state['processed_df'] = processed_df
    st.session_state['target'] = target

    st.success("Data preprocessing complete! Component 1 is ready for downstream steps.")

    # 3: Bar Charts
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

    # 4: Train Regression Model
    st.header("4. Train Regression Model")

    st.subheader("Select Features to Train On")
    available_features = [col for col in processed_df.columns if col != target]
    selected_features = st.multiselect("Choose features:", available_features)

    if selected_features:
        X = processed_df[selected_features]
        y = processed_df[target]

        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='mean'), numerical_cols),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_cols)
            ]
        )

        model = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('regressor', Ridge())
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        r2_score = model.score(X_test, y_test)

        st.session_state['model'] = model
        st.session_state['selected_features'] = selected_features

        st.success(f"Model trained successfully. R² score on test set: **{r2_score:.3f}**")
    else:
        st.warning("Please select at least one feature to train the model.")

    # 5: Prediction
    st.header("5. Predict Target Value")

    if 'model' not in st.session_state or 'selected_features' not in st.session_state:
        st.info("Please train the model first in step 4 to enable prediction.")
    else:
        user_input = st.text_input(
            f"Enter values for the selected features ({', '.join(st.session_state['selected_features'])})",
            placeholder="e.g., 0.5,1,0,23.1"
        )

        if st.button("Predict"):
            try:
                input_list = [float(x.strip()) for x in user_input.split(",")]
                if len(input_list) != len(st.session_state['selected_features']):
                    st.error(f"Expected {len(st.session_state['selected_features'])} values, but got {len(input_list)}.")
                else:
                    input_df = pd.DataFrame([input_list], columns=st.session_state['selected_features'])
                    prediction = st.session_state['model'].predict(input_df)[0]
                    st.success(f"Predicted target value: **{prediction:.2f}**")
            except ValueError:
                st.error("Please enter only numbers, separated by commas.")
            st.write("You entered:", input_df)

if __name__ == '__main__':
    main()