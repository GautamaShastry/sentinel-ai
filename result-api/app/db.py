import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel")

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def fetch_all(query: str, params=None):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params or ())
            return cur.fetchall()

def fetch_one(query: str, params=None):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params or ())
            return cur.fetchone()

def execute(query: str, params=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            conn.commit()
