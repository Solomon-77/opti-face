import psycopg2

from dotenv import load_dotenv
import os

load_dotenv()

def connect_to_db():
    """
    Create a connection to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),       
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            port=os.getenv("DB_PORT")        
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None
