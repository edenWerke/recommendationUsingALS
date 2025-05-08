import psycopg2

try:
    conn = psycopg2.connect(
        user="postgres",
        password="edenw",
        host="localhost",
        port="5000",  # Change to 5432 if that's your default
        database="myDump"
    )
    print("Successfully connected to PostgreSQL database!")
    conn.close()
except Exception as e:
    print("Failed to connect to PostgreSQL database.")
    print("Error:", e)