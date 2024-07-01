from pathlib import Path
import sys
import yaml
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model(train_features, target, n_estimators, max_depth, seed):
    # Train your machine learning model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    model.fit(train_features, target)
    return model

def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path / 'model.joblib')
    return None

def main() -> None:
    """main function to call other functions"""
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1]
    data_path = home_dir / input_file
    output_path = home_dir / '/models'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    TARGET = 'trip_duration'
    train_features = pd.read_csv(data_path / '/train.csv')
    X = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]


    params_file = home_dir / '/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trained_model = train_model(X_train, y_train, n_estimators=params.n_estimators, max_depth=params.max_depth, seed=params.seed)
    save_model(trained_model, output_path)





if __name__ == "__main__":
    main()