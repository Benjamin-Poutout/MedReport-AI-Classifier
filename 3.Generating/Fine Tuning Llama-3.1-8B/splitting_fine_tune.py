import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Lire les données du fichier JSONL
with open("output_file.jsonl", "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

# Convertir les données en DataFrame
df = pd.DataFrame(data)

# Diviser les données en ensembles d'entraînement et de test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Sauvegarder les ensembles d'entraînement et de test dans de nouveaux fichiers JSONL
train_df.to_json("llama_3_finetune_data_train.jsonl", orient="records", lines=True)
test_df.to_json("llama_3_finetune_data_test.jsonl", orient="records", lines=True)

print("Les fichiers 'output_train.jsonl' et 'output_test.jsonl' ont été créés avec succès.")
