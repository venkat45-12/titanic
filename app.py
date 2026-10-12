"""Streamlit app. Run: streamlit run app.py"""
import json
import joblib
import streamlit as st
from predict import make_passenger

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")


@st.cache_resource
def load():
    return joblib.load("models/titanic_model.joblib"), json.load(open("models/metadata.json"))


model, meta = load()
st.caption(f"Model: {meta['best_model']} | Test accuracy: {meta['test_accuracy']:.1%} "
           f"| ROC-AUC: {meta['roc_auc']:.3f}")

c1, c2 = st.columns(2)
sex = c1.selectbox("Sex", ["male", "female"])
pclass = c1.selectbox("Ticket class", [1, 2, 3], index=2)
age = c1.slider("Age", 1, 80, 30)
embarked = c1.selectbox("Port of embarkation", ["S", "C", "Q"],
                        format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
fare = c2.number_input("Fare (£)", 0.0, 520.0, 15.0)
sibsp = c2.number_input("Siblings / spouses aboard", 0, 8, 0)
parch = c2.number_input("Parents / children aboard", 0, 6, 0)
title = c2.selectbox("Title", ["Mr", "Mrs", "Miss", "Master"], index=0 if sex == "male" else 1)

if st.button("Predict", type="primary"):
    prob = model.predict_proba(make_passenger(pclass, sex, age, sibsp, parch, fare, embarked, title))[0, 1]
    st.metric("Survival probability", f"{prob:.1%}")
    st.progress(float(prob))
    (st.success if prob >= 0.5 else st.error)("Likely survived" if prob >= 0.5 else "Likely did not survive")
