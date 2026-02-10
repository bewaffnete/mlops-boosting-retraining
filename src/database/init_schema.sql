CREATE TABLE IF NOT EXISTS train (
    id SERIAL PRIMARY KEY,
    passenger_count INTEGER,
    pickup_longitude DOUBLE PRECISION,
    pickup_latitude DOUBLE PRECISION,
    dropoff_longitude DOUBLE PRECISION,
    dropoff_latitude DOUBLE PRECISION,
    N BOOLEAN,
    Y BOOLEAN,
    field_1 BOOLEAN,
    field_2 BOOLEAN,
    month INTEGER,
    week INTEGER,
    weekday INTEGER,
    hour INTEGER,
    minute_oftheday INTEGER,
    distance DOUBLE PRECISION,
    direction DOUBLE PRECISION,
    speed DOUBLE PRECISION,
    trip_duration DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS test (LIKE train INCLUDING ALL);

CREATE TABLE IF NOT EXISTS new_data (
    id SERIAL PRIMARY KEY,
    vendor_id INTEGER,
    pickup_datetime TEXT,
    passenger_count INTEGER,
    pickup_longitude DOUBLE PRECISION,
    pickup_latitude DOUBLE PRECISION,
    dropoff_longitude DOUBLE PRECISION,
    dropoff_latitude DOUBLE PRECISION,
    store_and_fwd_flag TEXT,
    -- Get via API
    trip_duration DOUBLE PRECISION
);

