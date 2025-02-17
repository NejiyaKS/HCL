import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

model = joblib.load('credit_risk_model.pkl')

df = pd.read_csv('bankloans.csv')

st.title("Credit Risk Analysis System")

st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.slider("Age", int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
    ed = st.sidebar.slider("Education", int(df['ed'].min()), int(df['ed'].max()), int(df['ed'].mean()))
    employ = st.sidebar.slider("Employment Years", int(df['employ'].min()), int(df['employ'].max()), int(df['employ'].mean()))
    address = st.sidebar.slider("Address Years", int(df['address'].min()), int(df['address'].max()), int(df['address'].mean()))
    income = st.sidebar.slider("Income", int(df['income'].min()), int(df['income'].max()), int(df['income'].mean()))
    debtinc = st.sidebar.slider("Debt to Income Ratio", int(df['debtinc'].min()), int(df['debtinc'].max()), int(df['debtinc'].mean()))
    creddebt = st.sidebar.slider("Credit Debt", int(df['creddebt'].min()), int(df['creddebt'].max()), int(df['creddebt'].mean()))
    othdebt = st.sidebar.slider("Other Debt", int(df['othdebt'].min()), int(df['othdebt'].max()), int(df['othdebt'].mean()))
    
    data = {
        'age': age,
        'ed': ed,
        'employ': employ,
        'address': address,
        'income': income,
        'debtinc': debtinc,
        'creddebt': creddebt,
        'othdebt': othdebt,
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

st.subheader("User Input Features")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)
st.subheader("Prediction")
risk_labels = ["Low Risk", "High Risk"]
st.write(f"**Prediction:** {risk_labels[int(prediction[0])]}")
st.write(f"**Prediction Probability:** {prediction_proba[0]}")

st.subheader("Data Visualizations")

st.write("**Feature Importance**")
feature_importance = pd.Series(model.feature_importances_, index=df.drop('default', axis=1).columns)
st.bar_chart(feature_importance.sort_values(ascending=False))

st.write("**Correlation Heatmap**")
corr = df.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)