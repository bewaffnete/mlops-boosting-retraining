import pandas as pd
import numpy as np
from data_processing.schemas import ModelData
from pydantic import ValidationError
from database.main import erase, insert_transaction

AVG_EARTH_RADIUS = 6371


def ft_haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def ft_degree(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    valid_rows = []
    valid_indices = []
    for i, row in df.iterrows():
        try:
            user = ModelData(**row.to_dict())
            valid_rows.append(user)
            valid_indices.append(i)
        except ValidationError:
            continue
    return pd.DataFrame([u.model_dump() for u in valid_rows]), valid_indices


def transform(df: pd.DataFrame):
    df = pd.concat([df, pd.get_dummies(df['store_and_fwd_flag'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['vendor_id'])], axis=1)

    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)

    df['month'] = df.pickup_datetime.dt.month
    df['week'] = df.pickup_datetime.dt.isocalendar().week
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour'] = df.pickup_datetime.dt.hour
    df['minute'] = df.pickup_datetime.dt.minute
    df['minute_oftheday'] = df['hour'] * 60 + df['minute']
    cols_to_int64 = ['month', 'week', 'weekday', 'hour', 'minute', 'minute_oftheday']

    df[cols_to_int64] = df[cols_to_int64].astype('int64')

    df['distance'] = ft_haversine_distance(df['pickup_latitude'].values,
                                           df['pickup_longitude'].values,
                                           df['dropoff_latitude'].values,
                                           df['dropoff_longitude'].values)

    df['direction'] = ft_degree(df['pickup_latitude'].values,
                                df['pickup_longitude'].values,
                                df['dropoff_latitude'].values,
                                df['dropoff_longitude'].values)

    df['trip_duration'] = np.log1p(df['trip_duration'].values)
    df['speed'] = df.distance / df.trip_duration

    df.drop(columns=['store_and_fwd_flag',
                     'vendor_id',
                     'pickup_datetime',
                     'minute'], axis=1, inplace=True)

    df.rename(columns={1: 'field_1', 2: 'field_2', 'N': 'n', 'Y': 'y'}, inplace=True)

    id_series = df['id'] if 'id' in df.columns else None
    if id_series is not None:
        df = df.drop(columns=['id'])

    df, valid_indices = validate(df)

    for index, row in df.iterrows():
        data_dict = row[df.columns].to_dict()
        insert_transaction(data_dict, 'train')

    if id_series is not None and valid_indices:
        valid_ids = id_series.iloc[valid_indices].tolist()
        erase(pd.DataFrame({"id": valid_ids}))

    return df


if __name__ == '__main__':
    pass
