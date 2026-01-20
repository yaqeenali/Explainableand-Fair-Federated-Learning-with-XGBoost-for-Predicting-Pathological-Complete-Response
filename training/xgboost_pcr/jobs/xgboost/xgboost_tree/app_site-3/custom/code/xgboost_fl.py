# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

import argparse
import csv
import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nvflare import client as flare
from nvflare.app_opt.xgboost.tree_based.shareable_generator import update_model


def to_dataset_tuple(data: dict):
    dataset_tuples = {}
    for dataset_name, dataset in data.items():
        dataset_tuples[dataset_name] = _to_data_tuple(dataset)
    return dataset_tuples

def _to_data_tuple(data):
    data_num = data.shape[0]
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return x.to_numpy(), y.to_numpy(), data_num

def load_features(feature_data_path: str) -> List:
    try:
        with open(feature_data_path, "r") as file:
            csv_reader = csv.reader(file)
            return next(csv_reader)
    except Exception as e:
        raise Exception(f"Load header for path '{feature_data_path}' failed! {e}")

def load_data(data_path: str, data_features: List, random_state: int, test_size: float, skip_rows=None) -> Dict[str, pd.DataFrame]:
    try:
        df: pd.DataFrame = pd.read_csv(
            data_path, names=data_features, sep=r"\s*,\s*", engine="python", na_values="?", skiprows=skip_rows
        )
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
        return {"train": train, "test": test}
    except Exception as e:
        raise Exception(f"Load data for path '{data_path}' failed! {e}")

def transform_data(data: Dict[str, Tuple]) -> Dict[str, Tuple]:
    scaler = StandardScaler()
    scaled_datasets = {}
    for dataset_name, (x_data, y_data, data_num) in data.items():
        x_scaled = scaler.fit_transform(x_data)
        scaled_datasets[dataset_name] = (x_scaled, y_data, data_num)
    return scaled_datasets

def evaluate_model(x_test, model, y_test):
    dtest = xgb.DMatrix(x_test)
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_test_bin = (y_test > 0.5).astype(int)
    auc = roc_auc_score(y_test_bin, y_pred_prob)
    bal_acc = balanced_accuracy_score(y_test_bin, y_pred)
    return auc, bal_acc

def define_args_parser():
    parser = argparse.ArgumentParser(description="XGBoost training client for NVFLARE")
    parser.add_argument("--data_root_dir", type=str, help="Root directory path to CSV data files")
    parser.add_argument("--random_state", type=int, default=0, help="Random state")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test ratio")
    parser.add_argument("--num_client_bagging", type=int, default=4, help="Number of clients for bagging")
    parser.add_argument("--skip_rows", type=str, default=[], help="Rows to skip when reading CSV")
    return parser

def main():
    parser = define_args_parser()
    args = parser.parse_args()
    data_root_dir = args.data_root_dir
    random_state = args.random_state
    test_size = args.test_size
    skip_rows = args.skip_rows
    num_client_bagging = args.num_client_bagging

    flare.init()
    site_name = flare.get_site_name()
    feature_data_path = f"{data_root_dir}/{site_name}_header.csv"
    features = load_features(feature_data_path)
    n_features = len(features) - 1

    data_path = f"{data_root_dir}/{site_name}.csv"
    data = load_data(data_path=data_path, data_features=features, random_state=random_state, test_size=test_size, skip_rows=skip_rows)
    data = to_dataset_tuple(data)
    dataset = transform_data(data)
    x_train, y_train, train_size = dataset["train"]
    x_test, y_test, test_size = dataset["test"]

    dmat_train = xgb.DMatrix(x_train, label=y_train)
    dmat_test = xgb.DMatrix(x_test, label=y_test)

    xgb_params = {
        "eta": 0.1,
        "objective": "binary:logistic",
        "max_depth": 3,
        "eval_metric": "auc",
        #"n_estimators": 300,
        "nthread": 16,
        "num_parallel_tree": 1,
        "subsample": 1.0,
        "tree_method": "hist",
        "colsample_bytree": 0.6,
        "gamma": 0,
        "min_child_weight": 3,
        "scale_pos_weight": 3.6
    }

    global_model_as_dict = None
    auc = 0.0
    bal_acc = 0.0
    best_bal_acc = 0.0
    best_auc = 0.0
    best_model = None
    config = None

    while flare.is_running():
        input_model = flare.receive()
        global_params = input_model.params
        curr_round = input_model.current_round
        print(f"current_round={curr_round}")

        if curr_round == 0:
            num_boost_round = 1
            model = None

            for i in range(num_boost_round):
                if model is None:
                    model = xgb.train(
                        xgb_params,
                        dmat_train,
                        num_boost_round=1,
                        evals=[(dmat_train, "train"), (dmat_test, "test")],
                        verbose_eval=False,
                    )
                    config = model.save_config()
                else:
                    model.update(dmat_train, i)

                auc, bal_acc = evaluate_model(x_test, model, y_test)
                print(f"{site_name} | Round 0 | Iter {i+1} | AUC: {auc:.5f} | Balanced Acc: {bal_acc:.5f}")

                if bal_acc > best_bal_acc and auc > best_auc:
                    best_bal_acc = bal_acc
                    best_auc = auc
                    best_model = model.copy()
                    print(f">>> New best model found and saved (AUC={auc:.5f}, BalAcc={bal_acc:.5f})")

            model = best_model
            model.load_config(config)
            model.save_model(f"{site_name}_best_model.json")

        else:
            model_updates = global_params["model_data"]
            for update in model_updates:
                global_model_as_dict = update_model(global_model_as_dict, json.loads(update))
            loadable_model = bytearray(json.dumps(global_model_as_dict), "utf-8")

            if model is None:
                model = xgb.Booster()
            model.load_model(loadable_model)
            model.load_config(config)

            auc, bal_acc = evaluate_model(x_test, model, y_test)
            print(f"{site_name}: Global model AUC: {auc:.5f} | Balanced Acc: {bal_acc:.5f}")

            if bal_acc > best_bal_acc and auc > best_auc:
                best_bal_acc = bal_acc
                best_auc = auc
                best_model = model.copy()
                best_model.save_model(f"{site_name}_best_model.json")
                print(f">>> New best model saved (AUC={auc:.5f}, BalAcc={bal_acc:.5f})")

            model.update(dmat_train, model.num_boosted_rounds())

        bst_new = model[model.num_boosted_rounds() - 1 : model.num_boosted_rounds()]
        local_model_update = bst_new.save_raw("json")
        params = {"model_data": local_model_update}
        metrics = {"auc": auc, "balanced_accuracy": bal_acc}
        output_model = flare.FLModel(params=params, metrics=metrics)
        flare.send(output_model)

if __name__ == "__main__":
    main()

