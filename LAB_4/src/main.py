import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from decisiontree import DecisionTree
from randomforest import RandomForest
from evaluate import f1_macro

def load_data(path):
    df = pd.read_csv(path, sep=";")
    X = df.drop("quality", axis=1).values
    y = df["quality"].values
    return X, y

DATA_PATH = 'data/winequality-red.csv'

def assignment_1():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTree(max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = f1_macro(y_test, y_pred)
    print(f"[Assignment 1] Decision Tree (NumPy) — Macro F1: {score:.4f}")
    return score

def assignment_2():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForest(n_estimators=50, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = f1_macro(y_test, y_pred)
    print(f"[Assignment 2] Random Forest (NumPy) — Macro F1: {score:.4f}")
    return score


def assignment_3():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_score = f1_score(y_test, dt.predict(X_test), average='macro')
    print(f"[Assignment 3] Decision Tree (sklearn) — Macro F1: {dt_score:.4f}")

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_score = f1_score(y_test, rf.predict(X_test), average='macro')
    print(f"[Assignment 3] Random Forest (sklearn) — Macro F1: {rf_score:.4f}")

    return dt_score, rf_score

if __name__ == '__main__':
    assignment_1()
    assignment_2()
    assignment_3()
