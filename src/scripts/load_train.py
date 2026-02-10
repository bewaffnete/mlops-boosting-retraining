import argparse
import os
from typing import Dict

import pandas as pd
from sqlalchemy import create_engine

from config import backend


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    if "1" in df.columns:
        rename_map["1"] = "field_1"
    if "2" in df.columns:
        rename_map["2"] = "field_2"
    if "n" in df.columns and "N" not in df.columns:
        rename_map["n"] = "N"
    if "y" in df.columns and "Y" not in df.columns:
        rename_map["y"] = "Y"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _drop_index_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("id", "Unnamed: 0"):
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def load_xy(x_path: str, y_path: str, table: str) -> None:
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    x_df = _drop_index_cols(_normalize_columns(x_df))
    y_df = _drop_index_cols(y_df)

    if y_df.shape[1] == 1 and "trip_duration" not in y_df.columns:
        y_df = y_df.rename(columns={y_df.columns[0]: "trip_duration"})

    df = pd.concat([x_df, y_df], axis=1)

    engine = create_engine(backend)
    df.to_sql(table, engine, if_exists="append", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", default=os.getenv("X_PATH", "data/X_train.zip"))
    parser.add_argument("--y", default=os.getenv("Y_PATH", "data/y_train.zip"))
    parser.add_argument("--table", default=os.getenv("TABLE_NAME", "train"))
    args = parser.parse_args()

    load_xy(args.x, args.y, args.table)


if __name__ == "__main__":
    main()
