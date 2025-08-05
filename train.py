import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os
import argparse

def train_and_evaluate(data_path):
    MODEL_DIR = 'artifacts'
    MODEL_NAME = 'model.joblib'
    METRICS_FILE = 'metrics.txt'
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading data from {data_path}...")
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        exit(1)

    train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train.species
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test.species

    mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
    mod_dt.fit(X_train, y_train)

    prediction = mod_dt.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    print(f'The accuracy of the Decision Tree is: {accuracy:.3f}')

    print(f"Saving metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, "w") as f:
        f.write(f"Accuracy: {accuracy:.3f}\n")

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(mod_dt, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help="Path to the CSV dataset")
    args = parser.parse_args()
    train_and_evaluate(args.data_path)
