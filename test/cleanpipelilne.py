
import pandas as pd
import json
import re
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

INPUT_FILE = r"C:\Users\Harshit Sharma\OneDrive\Desktop\New folder\merged_data.csv"

# Save output beside script
BASE_DIR = Path(__file__).parent

OUTPUT_FILE = BASE_DIR / "train.jsonl"

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading dataset...")

try:
    df = pd.read_csv(INPUT_FILE)

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

print("Dataset loaded successfully.")
print(f"Total rows: {len(df)}")

# =====================================================
# CLEAN SCHEMA FUNCTION
# =====================================================

# Example:
# {'table': 'employees', 'columns': ['id', 'name']}
# -->
# employees(id, name)

def clean_schema(schema_text):

    try:
        table_match = re.search(
            r"'table':\s*'([^']+)'",
            schema_text
        )

        columns_match = re.search(
            r"'columns':\s*\[(.*?)\]",
            schema_text
        )

        table_name = (
            table_match.group(1)
            if table_match
            else "table"
        )

        columns_raw = (
            columns_match.group(1)
            if columns_match
            else ""
        )

        columns = re.findall(
            r"'([^']+)'",
            columns_raw
        )

        cleaned_schema = (
            f"{table_name}({', '.join(columns)})"
        )

        return cleaned_schema

    except Exception:
        return "table(column1, column2)"

# =====================================================
# CLEAN SQL FUNCTION
# =====================================================

def clean_sql(sql):

    sql = str(sql).strip()

    # Add semicolon if missing
    if not sql.endswith(";"):
        sql += ";"

    return sql

# =====================================================
# BUILD TRAINING PROMPT
# =====================================================

def build_prompt(schema, instruction, query):

    prompt = (
        "<s>[INST] Generate SQL query only.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question:\n{instruction}\n\n"
        "[/INST]\n"
        f"{query}</s>"
    )

    return prompt

# =====================================================
# PROCESS DATASET
# =====================================================

processed_rows = []

print("\nProcessing dataset...\n")

for index, row in df.iterrows():

    try:

        instruction = str(
            row["instruction"]
        ).strip()

        query = clean_sql(
            row["query"]
        )

        schema = clean_schema(
            str(row["database"])
        )

        sample = {
            "text": build_prompt(
                schema,
                instruction,
                query
            )
        }

        processed_rows.append(sample)

        # Print first 3 samples
        if index < 3:

            print("=" * 80)
            print(f"SAMPLE {index + 1}")
            print("=" * 80)

            print(
                json.dumps(
                    sample,
                    indent=2,
                    ensure_ascii=False
                )
            )

            print()

    except Exception as e:

        print(
            f"Error processing row {index}: {e}"
        )

# =====================================================
# SAVE JSONL FILE
# =====================================================

print("=" * 80)
print("Saving JSONL file...")
print("=" * 80)

print(f"Output path: {OUTPUT_FILE}")

try:

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for row in processed_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\nJSONL file saved successfully.")
    print(f"Saved rows: {len(processed_rows)}")

except Exception as e:

    print(f"Error saving JSONL: {e}")

# =====================================================
# VALIDATION
# =====================================================

print("\nValidation Check")
print("=" * 80)

try:

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        first_line = f.readline()

    print("First JSONL Row:\n")
    print(first_line)

except Exception as e:

    print(f"Validation Error: {e}")

# =====================================================
# FINISHED
# =====================================================

print("\nPipeline completed successfully.")

