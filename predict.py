"""Predict survival for one passenger.

Example:
  python predict.py --pclass 1 --sex female --age 29 --sibsp 0 --parch 0 --fare 80 --embarked S
"""
import argparse
import joblib
import pandas as pd

DEFAULT_TITLE = {"male": "Mr", "female": "Mrs"}


def make_passenger(pclass, sex, age, sibsp, parch, fare, embarked, title=None, cabin=None):
    title = title or DEFAULT_TITLE[sex]
    return pd.DataFrame([{
        "Pclass": pclass, "Name": f"Doe, {title}. John", "Sex": sex, "Age": age,
        "SibSp": sibsp, "Parch": parch, "Fare": fare, "Cabin": cabin, "Embarked": embarked,
    }])


def main():
    p = argparse.ArgumentParser(description="Titanic survival predictor")
    p.add_argument("--pclass", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--sex", choices=["male", "female"], required=True)
    p.add_argument("--age", type=float, required=True)
    p.add_argument("--sibsp", type=int, default=0, help="siblings/spouses aboard")
    p.add_argument("--parch", type=int, default=0, help="parents/children aboard")
    p.add_argument("--fare", type=float, required=True)
    p.add_argument("--embarked", choices=["S", "C", "Q"], default="S")
    p.add_argument("--title", choices=["Mr", "Mrs", "Miss", "Master"], default=None)
    a = p.parse_args()

    model = joblib.load("models/titanic_model.joblib")
    row = make_passenger(a.pclass, a.sex, a.age, a.sibsp, a.parch, a.fare, a.embarked, a.title)
    prob = model.predict_proba(row)[0, 1]
    print(f"Survival probability: {prob:.1%}")
    print("Prediction:", "SURVIVED" if prob >= 0.5 else "DID NOT SURVIVE")


if __name__ == "__main__":
    main()
