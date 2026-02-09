from dotenv import load_dotenv
import os

load_dotenv()

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")

OPTUNA = os.getenv("USING_OPTUNA")

postgres_user = os.getenv('PG_USER')
postgres_password = os.getenv('PG_PASSWORD')
postgres_host = os.getenv('PG_HOST', 'localhost')
postgres_port = os.getenv('PG_PORT', '5432')
postgres_db = os.getenv('PG_DB')

backend = f'postgresql+psycopg2://{postgres_user}:{postgres_password}@localhost:{postgres_port}/{postgres_db}'

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "db"),
    "port": int(os.getenv("PG_PORT", 5432)),
    "dbname": os.getenv("PG_DB"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
}
