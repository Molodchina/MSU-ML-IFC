from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor


def shape_df(df):
    df = df.drop(columns=["Unnamed: 0"])

    # print(df.dtypes)
    categorical_columns = ["cut", "color", "clarity"]
    encoder = OneHotEncoder(sparse_output=False)
    oh_encoder = encoder.fit_transform(df[categorical_columns])
    one_hot_df = pd.DataFrame(oh_encoder,
                              columns=encoder.get_feature_names_out(categorical_columns))
    df = pd.concat([df.drop(categorical_columns, axis=1), one_hot_df], axis=1)

    df = shuffle(df, random_state=42)
    return df


dataset_name: str = "train6.csv"
dataset_dir: Path = Path("datasets")
dataset_path: Path = dataset_dir.joinpath(dataset_name)
df = shape_df(pd.read_csv(dataset_path))

y = df['price']
X = df.drop(columns=['price'])

params = [
    {'criterion': 'squared_error', 'max_depth': 12},
    {'criterion': 'friedman_mse', 'max_depth': 16},
    {'criterion': 'poisson', 'max_depth': 22},
    {'criterion': 'squared_error', 'max_depth': 45},
    {'criterion': 'friedman_mse', 'max_depth': 95},
    {'criterion': 'poisson', 'max_depth': 33}
]

best_param = None
best_score = -float('inf')
for param in params:
    clf = DecisionTreeRegressor(**param)
    scores = cross_val_score(clf, X, y, cv=10, scoring='r2')
    mean_score = np.mean(scores)
    print(f"Params: {param}, Mean CV Score: {mean_score}")

    if mean_score > best_score:
        best_score = mean_score
        best_param = param
print(f"Best model params: {best_param}, Best MSE: {best_score}")
