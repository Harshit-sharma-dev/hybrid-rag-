# Generate SQL
sql_query = ask_llama(prompt)

# Clean SQL
sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

# Execute
df = pd.read_sql(sql_query, engine)

# Convert result
result_text = df.to_string(index=False)

# Ask LLM for final answer
final_prompt = f"""
Question: {question}
Result: {result_text}

Give only the final answer.
"""

final_answer = ask_llama(final_prompt)

print(final_answer)