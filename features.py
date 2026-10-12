"""Shared feature engineering. Imported by train.py, predict.py and app.py."""
import pandas as pd

RARE_TITLES = {"Dr", "Rev", "Col", "Major", "Capt", "Sir", "Lady", "Don",
               "Dona", "Jonkheer", "Countess"}
NUMERIC = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone"]
RAW_COLUMNS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw passenger columns into model-ready columns."""
    df = df.copy()
    title = df["Name"].str.extract(r",\s*[^.]*?\b([A-Za-z]+)\.", expand=False)
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = title.where(~title.isin(RARE_TITLES), "Rare").fillna("Unknown")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int).astype(str)
    df["Deck"] = df["Cabin"].fillna("U").astype(str).str[0]
    df["Pclass"] = df["Pclass"].astype(str)
    return df[NUMERIC + CATEGORICAL]
