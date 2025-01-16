import os
import psycopg2
from psycopg2 import OperationalError, InterfaceError
from dotenv import load_dotenv
import time

# Load environment variables from the .env file
load_dotenv()

# Get the PostgreSQL credentials from environment variables
pg_host = os.getenv("PG_HOST")
pg_dbname = os.getenv("PG_DBNAME")
pg_username = os.getenv("PG_USERNAME")
pg_password = os.getenv("PG_PASSWORD")

def get_db_connection(retries=5, delay=5):
    """Creates and returns a new database connection with retry logic."""
    attempt = 0
    while attempt < retries:
        try:
            return psycopg2.connect(
                dbname=pg_dbname,
                user=pg_username,
                password=pg_password,
                host=pg_host
            )
        except OperationalError as e:
            print(f"Database connection failed: {e}. Retrying in {delay} seconds...")
            attempt += 1
            time.sleep(delay)
    
    raise Exception("Failed to connect to the database after multiple attempts.")

# Initial connection
connection = get_db_connection()

def ensure_connection():
    """Checks if the current connection is alive, and reconnects if it is not."""
    global connection
    try:
        # Attempt to use the connection to ensure it is still alive
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except (OperationalError, InterfaceError):
        # If an OperationalError or InterfaceError is caught, reconnect
        print("Connection lost. Reconnecting...")
        connection = get_db_connection()
