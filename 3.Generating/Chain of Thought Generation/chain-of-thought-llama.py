import json
import numpy as np
import csv
from transformers import BertTokenizer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time
from accelerate import Accelerator

def normalize_case_data(case_data):
    normalized_data = {}
    for key, value in case_data.items():
        if isinstance(value, str):
            normalized_data[key] = [value]
        elif isinstance(value, list):
            normalized_data[key] = value
        else:
            normalized_data[key] = ["N/A"]
    return normalized_data

def get_first_three_words(text):
    words = text.split()
    return ' '.join(words[:3])

accelerator = Accelerator()

model_add = 'meta-llama/Meta-Llama-3-70B-Instruct'
access_token = "hugging_face_token_here"

# Initializing the model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_add, use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_add, use_auth_token=access_token, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model.eval()

exemples = []
with open('Entities_to_Generate.csv', newline='', encoding='utf-8') as f:
    cases = list(csv.DictReader(f))

    # Limiting to 5 cases for the examples
    for case_index, case in enumerate(cases):
        if case_index >= 5:
            break

        case_to_generate = case.get('generation', 'N/A')
        real_text = case.get('true_text', 'N/A')

        exemples.append(f"""
        Exemple {case_index + 1} :
        Entités Extraites :
        {case_to_generate}

        Vrai Texte :
        {real_text}
        """)

exemples_str = "\n".join(exemples)

# List to store the time taken by generation
avg_time=[]

# Opening the CSV file to write results
with open('chain_of_thought_llama_70B.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['model_name', 'prompt', 'generation', 'true_text']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Générating new items and adding them to CSV file
    j=0
    for case_index, case in enumerate(cases):
        print(f"Number of GPUs available : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1073741824:.2f} GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i)/1073741824:.2f} GB")
        torch.cuda.empty_cache()
        start_time = time.time()
        j+=1
        cas_a_generer = case.get('generation', 'N/A')
        tokens = tokenizer.encode(cas_a_generer)
        token_count = len(tokens)

        start_sentence = get_first_three_words(case.get('true_text', 'N/A'))

        prompt = f"""Ton objectif est de rédiger un rapport de cas médical en te basant sur les entités fournies. Adopte une réflexion méthodique, étape par étape, comme le ferait un médecin spécialisé dans la rédaction de rapports médicaux. Analyse attentivement les informations contenues dans les entités extraites, en veillant à respecter la chronologie des événements et à aborder chaque élément de manière logique. Une fois chaque aspect bien compris, rédige le rapport de cas médical de manière concise et professionnelle, dans un seul paragraphe, en adoptant le style et le ton d'un médecin spécialiste."""

        user_prompt = f"""Taille du cas à générer {token_count} tokens.

        Démarre le texte avec les mots {start_sentence}.

        Entités Extraites pour le Cas à Générer :
        {cas_a_generer}
        """
        print(start_sentence)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(accelerator.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.15,
            top_k=20,
            top_p=0.9,
        )
        del input_ids
        response = tokenizer.decode(outputs[0])
        end_time = time.time()
        duration = end_time - start_time
        avg_time.append(duration)
        print(f"Generating the sentence {j} took {duration} seconds", flush=True)
        print(response)
        print(duration)

        new_item = {
            'model_name': model_add,
            'prompt': prompt,
            'generation': response,
            'true_text': case.get('true_text', 'N/A')
        }

        writer.writerow(new_item)

# Average time taken
avg = sum(avg_time) / len(avg_time)
print(avg)
