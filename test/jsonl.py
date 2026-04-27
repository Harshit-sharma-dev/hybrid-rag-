import pandas as pd

# -----------------------------
# 1. Load Data
# -----------------------------
file_path = r"C:\Users\Harshit Sharma\OneDrive\Desktop\New folder\Financials.csv"

df = pd.read_csv(file_path)

# -----------------------------
# 2. Validate structure
# -----------------------------
if df.empty:
    raise ValueError("Dataset is empty")

# Ensure first column exists
first_col = df.columns[0]

# -----------------------------
# 3. Set index (row labels)
# -----------------------------
df = df.set_index(first_col)

# -----------------------------
# 4. Transpose (pivot → tidy)
# -----------------------------
df = df.T.reset_index(drop=True)

# -----------------------------
# 5. Clean currency safely
# -----------------------------
def clean_money(x):
    try:
        if pd.isna(x):
            return 0.0

        if isinstance(x, str):
            x = x.strip()
            if x in ["", "-", "$-"]:
                return 0.0

            x = x.replace("$", "").replace(",", "")
        
        return float(x)

    except Exception:
        return 0.0


money_cols = [
    'Units Sold', 'Manufacturing Price', 'Sale Price',
    'Gross Sales', 'Discounts', 'Sales', 'COGS', 'Profit'
]

# Apply only if column exists
for col in money_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_money)

# -----------------------------
# 6. Convert date safely
# -----------------------------
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

# -----------------------------
# 7. Optional: fix data types
# -----------------------------
numeric_cols = ['Month Number', 'Year']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# -----------------------------
# 8. Save to JSONL
# -----------------------------
output_path = "output.jsonl"
df.to_json(output_path, orient="records", lines=True, date_format="iso")

print(f"✅ Successfully saved to {output_path}")