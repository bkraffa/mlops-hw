import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-rf")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="../data",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str): 

    with mlflow.start_run():

        mlflow.set_tag("developer", "bruno")

        mlflow.log_param("train-data-path", "./data/green_tripdata_2022-01.parquet")
        mlflow.log_param("valid-data-path", "./data/green_tripdata_2022-02.parquet")
        mlflow.log_param("test-data-path", "./data/green_tripdata_2022-03.parquet")

        max_depth = 10
        mlflow.log_param("max_depth", max_depth)

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()