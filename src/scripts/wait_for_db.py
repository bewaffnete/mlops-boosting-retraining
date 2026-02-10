import os
import sys
import time
import psycopg


def main():
    host = os.getenv("PG_HOST", "db")
    port = int(os.getenv("PG_PORT", "5432"))
    dbname = os.getenv("PG_DB", "mlops")
    user = os.getenv("PG_USER", "mlops")
    password = os.getenv("PG_PASSWORD", "mlops")

    deadline = time.time() + 60
    last_err = None
    while time.time() < deadline:
        try:
            conn = psycopg.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
            )
            conn.close()
            break
        except Exception as e:
            last_err = e
            time.sleep(2)
    else:
        raise RuntimeError(f"DB not ready: {last_err}")

    if len(sys.argv) > 1:
        os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
