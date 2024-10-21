import json
import pandas as pd
from sklearn.model_selection import train_test_split

with open("output_file.jsonl", "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

df = pd.DataFrame(data)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_json("llama_3_finetune_data_train.jsonl", orient="records", lines=True)
test_df.to_json("llama_3_finetune_data_test.jsonl", orient="records", lines=True)

print("Th files 'output_train.jsonl' and 'output_test.jsonl' have been sucessfully created.")
