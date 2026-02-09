import pandas as pd
from pydantic import ValidationError
from data_processing.schemas import RawData
from database.main import insert_transaction

def main():
    df = pd.read_csv('data/inference.zip')
    for index, row in df.iterrows():
        data_dict = row[df.columns].to_dict()

        try:
            validated = RawData(**data_dict)
            insert_transaction(data_dict)
            # time.sleep(random.uniform(0.1, 1.5))
        except ValidationError:
            continue

if __name__ == '__main__':
    main()
