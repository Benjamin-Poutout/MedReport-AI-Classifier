import csv

llama_file = "llama_3_8B_five_shot_gen.csv"

with open(llama_file, 'r', newline='', encoding='utf-8-sig') as llama, \
     open('test.csv', 'w', newline='', encoding='utf-8-sig') as outfile:
    
    reader = csv.DictReader(llama)
    fieldnames = reader.fieldnames
    
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        generation_text = row['generation']
        
        # Check if "assistant" exist
        if "assistant<|end_header_id|>" in generation_text:
            # Exctracting everthing that comes after "assistant"
            text_after_assistant = generation_text.split("assistant<|end_header_id|>", 1)[1].strip()

            if "<|eot_id|>" in text_after_assistant:
                text_before_eot = text_after_assistant.split("<|eot_id|>", 1)[0].strip()
                row['generation'] = text_before_eot
            else:
                row['generation'] = text_after_assistant
            if "Voici le rapport de cas médical :" in text_after_assistant:
                text_final = text_before_eot.split("Voici le rapport de cas médical :", 1)[1].strip()
                row['generation'] = text_final
        
        writer.writerow(row)

print("Extraction terminée et fichier sauvegardé.")
