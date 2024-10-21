import json
import numpy as np
import csv
from transformers import BertTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time
from accelerate import Accelerator

accelerator = Accelerator()

with open('inference_data.json', 'r') as f:
    data = json.load(f)

# We create the sentences we will use in our SFT :

sentences = []
for item in data:
    if item['next_section']['text'] != "":
        sentences.append(item['next_section']['text'])
    for subsection in item['next_section']['subsections']:
        title = subsection['title']
        if title == "Patient":
            sentences.append(subsection['text'])

print(f"The dataset contains {len(sentences)} case reports.")

# Now, let's build our inference prompt and structure :

#bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model and the tokenizer :

model_add = 'meta-llama/Meta-Llama-3-70B-Instruct'
access_token = "hugging_face_token_here"

tokenizer = AutoTokenizer.from_pretrained(model_add, use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_add, use_auth_token=access_token, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

# Prompt to extract entities :

test = (f'''Vous êtes un expert dans le domaine médical, spécialisé dans l'analyse et la rédaction de rapports de cas. Voici votre tâche : Identifier et extraire toutes les entités pertinentes du rapport de cas médical suivant. Ces entités peuvent avoir été initialement enregistrées dans le dossier de santé électronique (EHR) du patient, soit sous forme structurée soit en texte libre. Ces entités identifiées doivent permettre de reconstituer fidèlement le rapport de cas original. Réfléchissez étape par étape et structurez votre réponse, en accordant une attention particulière à l'ordre chronologique des événements, car cela est crucial pour une compréhension complète du cas.''')


# Open the csv file in writing mode
fich = 'entity_llama_3_70B_1400.csv'
with open(fich, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['model_name', 'prompt', 'temperature', 'generation', 'true_text']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    # Generating the new items and adding them to a new csv file
    print(fich)
    j=0
    for sentence in sentences[1300:1400]:
        j+=1
        messages = [
        {"role": "system", "content": test},
        {"role": "user", "content":  sentence},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to('cuda')

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        start_time = time.time()

        outputs = model.generate(
            input_ids,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]

        end_time = time.time()
        duration = end_time - start_time
        print(f"Generating sentence {j} took {duration} seconds", flush=True)
        print(tokenizer.decode(response, skip_special_tokens=True))

        new_item = {
            'model_name': model_add,
            'prompt': test,
            'generation': tokenizer.decode(response, skip_special_tokens=True),
            'true_text': sentence
        }

        writer.writerow(new_item)

print("Data was saved in 'llama3.csv'.")
