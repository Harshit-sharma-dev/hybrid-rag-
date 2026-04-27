import pandas as pd
import sqlite3

# Load your CSV and save it to SQLite
df = pd.read_csv(r"C:\Users\Harshit Sharma\OneDrive\Desktop\New folder\Financials.csv")
conn = sqlite3.connect("my_db.PostgreSQL")
df.to_sql("sales_raw", conn, if_exists="replace", index=False)
conn.close()