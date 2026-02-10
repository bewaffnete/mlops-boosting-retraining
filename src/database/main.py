import psycopg
from config import DB_CONFIG, backend
import pandas as pd
from sqlalchemy import create_engine, text

def query_form(data_dict: dict, table_name='new_data'):
    """
    Form query.
    """

    if not data_dict:
        raise ValueError("Empty dict")

    columns = list(data_dict.keys())
    columns_str = ', '.join(columns)

    placeholders = ', '.join(['%s'] * len(columns))

    sql_query = f"""
        INSERT INTO {table_name} ({columns_str})
        VALUES ({placeholders})
        RETURNING id;
    """
    values = tuple(data_dict.values())
    return sql_query, values


def get_connection():
    """
    Connecting to the database.
    """
    try:
        conn = psycopg.connect(**DB_CONFIG)
        return conn
    except psycopg.Error as e:
        raise RuntimeError(f"Connection error: {e}") from e


def insert_transaction(data_dict: dict, table_name: str = 'new_data'):
    """
    Dynamic insertion of a transaction.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            query, values = query_form(data_dict, table_name)
            cur.execute(query, values)
            transaction_id = cur.fetchone()[0]
        conn.commit()

    return transaction_id


def select(table_name: str):
    engine = create_engine(backend)

    query = f"SELECT * FROM {table_name}"

    df = pd.read_sql(query, engine)
    return df


def erase(df: pd.DataFrame):
    ids_to_delete = df['id'].to_list()
    engine = create_engine(backend)

    if ids_to_delete:
        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM new_data WHERE id IN :ids"),
                {"ids": tuple(ids_to_delete)}
            )
            conn.commit()


if __name__ == '__main__':
    df = select('new_data')
    print(df.head())
