from pathlib import Path
import pandas as pd
import numpy as np

from feature_definitions import feature_build

def load_data(data_path: str) -> pd.DataFrame:
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_data(train: pd.DataFrame, test: pd.DataFrame, output_path: Path) -> None:
    # Save the split datasets to the specified output path
    Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(Path(output_path) / '/train.csv', index=False)
    test.to_csv(Path(output_path) / '/test.csv', index=False)
    return None


if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    train_path = home_dir / '/data/raw/train.csv'
    test_path = home_dir / '/data/raw/test.csv'

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    output_path = home_dir / '/data/processed'

    train_data = feature_build(train_data, 'train-data')
    test_data = feature_build(test_data, 'test-data')

    do_not_use_for_training = ['id', 'pickup_datetime', 'dropoff_datetime', 'check_trip_duration', 'pickup_date', 'avg_speed_h', 'avg_speed_m', 'pickup_datetime_group']

    feature_names = [f for f in train_data.columns if f not in do_not_use_for_training]
    print('We have %i features.' % len(feature_names))

    train_data = train_data[feature_names]
    test_data = test_data[feature_names]

    save_data(train_data, test_data, output_path)