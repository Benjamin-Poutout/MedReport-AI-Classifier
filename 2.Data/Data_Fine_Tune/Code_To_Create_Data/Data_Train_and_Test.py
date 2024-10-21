import csv
import json

# Paths to your CSV input file and JSONL output file
csv_file_path = 'Entities_to_Generate.csv'
jsonl_file_path = 'output_file.jsonl'

# Open the CSV and JSONL files
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file, open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
    # Read the CSV file
    csv_reader = csv.DictReader(csv_file)

    # Loop through each row in the CSV
    for row in csv_reader:
        # Create the desired JSONL format for each row
        json_obj = {
            "text": f" ###  Instruction :\n Tu es un médecin spécialiste en écriture de rapports de cas médicaux. Tu es chargé d'écrire un rapport de cas médical détaillé et professionnel basé sur les informations fournies. \n ### Input :\n {row['generation']} ### Response : \n {row['true_text']}"
        }

        # Write the JSON object as a line to the JSONL file
        jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

print(f"Conversion completed. JSONL file saved to {jsonl_file_path}")
