
  

# Data Analysis & Prediction App

This project is a Streamlit-based web application that allows users to upload datasets, analyze features visually, train a regression model, and make predictions. Built as part of a milestone group assignment, the app provides a complete data analysis pipeline through an interactive UI.


## 🚀 Features


### 1. **Upload Component**

- Upload a CSV file.
- Preprocesses data using scaling and one-hot encoding.

  
### 2. **Target Selection**
- Dynamically detects numeric columns.
- Lets user choose the target variable (dependent variable).
  

### 3. **Data Visualization**
- Bar chart 1: Average target value grouped by a selected categorical feature.
- Bar chart 2: Absolute correlation of numeric features with the target variable.
  

### 4. **Model Training**

- Lets user choose features to include.
- Trains a Ridge regression model using pipelines.
- Handles missing data and categorical encoding.
- Displays R² score after training.
  
### 5. **Prediction Component**

- Accepts comma-separated feature values.
- Predicts the target variable based on the trained model.
- Input validation for user-friendly experience.
  

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Altair (for charts)

## 📁 File Structure

```
.
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
```

## 🌐 Deployment

The app is deployed on Streamlit Cloud and accessible at: 👉 [Your Deployment Link Here]
  

## 🧑‍🤝‍🧑 Team Members

- Member 1: Upload + Preprocessing

- Member 2: Data Visualization

- Member 3: Model Training

- Member 4: Prediction + Deployment

  

## ✅ How to Run Locally

  

```

git clone https://github.com/andrewbeshay25/Data-Analysis-and-Prediction-App.git

cd Data-Analysis-and-Prediction-App

pip install -r requirements.txt

streamlit run app.py

```

----------

  Thank you for checking it out!