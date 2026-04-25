import requests
import pandas as pd
from sqlalchemy import create_engine
import re

# 🔹 DB connection (CHANGE PASSWORD + DB NAME)
engine = create_engine("postgresql://postgres:possword@localhost:5432/postgres")

# 🔹 LLM function
def ask_llama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# 🔹 Question
question = "What is the highest sales?"

# 🔹 Prompt (STRICT)
prompt = f"""
PostgreSQL expert.

Table: sales_raw
Columns: segment, country, product, discount_band, units_sold, manufacturing_price, sale_price, gross_sales, discounts, sales, cogs, profit, date, month_number, month_name, year

Rules:
- Use only these columns
- Use table sales_raw
- Return only SQL (no markdown, no explanation)

Question: {question}
"""

# 🔹 Generate SQL
sql_query = ask_llama(prompt)

# 🔹 Clean SQL (IMPORTANT)
sql_query = re.sub(r"```sql|```", "", sql_query).strip()

print("Generated SQL:", sql_query)

# 🔹 Execute SQL
df = pd.read_sql(sql_query, engine)

# 🔹 Output result
if df.shape == (1, 1):
    print("Answer:", df.iloc[0, 0])
else:
    print(df)