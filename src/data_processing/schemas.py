from pydantic import BaseModel, Field


class RawData(BaseModel):
    vendor_id: int
    pickup_datetime: str
    passenger_count: int = Field(gt=0)
    pickup_longitude: float = Field(gt=-100)
    pickup_latitude: float = Field(lt=50)
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str

    trip_duration: float = Field(lt=6000)


class ModelData(BaseModel):
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    n: bool
    y: bool
    field_1: bool
    field_2: bool
    month: int
    week: int
    weekday: int
    hour: int
    minute_oftheday: int
    distance: float = Field(lt=200)
    direction: float
    speed: float

    trip_duration: float
