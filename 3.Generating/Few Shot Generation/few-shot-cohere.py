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

quant_config = BitsAndBytesConfig(load_in_4bit=True)

model_id = "CohereForAI/c4ai-command-r-plus-4bit"
access_token = "hugging_face_token_here" 

# Initializing the model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=access_token, device_map="auto", quantization_config=quant_config, torch_dtype="auto", attn_implementation="flash_attention_2")
model.eval()

exemples = []
with open('Entities_to_Generate.csv', newline='', encoding='utf-8') as f:
   cases = list(csv.DictReader(f))

   # Limiting to 5 cases for the examples
   for case_index, case in enumerate(cases):
       if case_index >= 5:
           break

       # Supposing 'generation' and 'true_text' are our columns
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

# List to stock the time taken by generation
avg_time=[]

with open('command_r_plus_4bit_few_shot_gen.csv', 'w', newline='', encoding='utf-8') as csvfile:
   fieldnames = ['model_name', 'prompt', 'generation', 'true_text']
   writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
   writer.writeheader()

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

       prompt = f"""Tu es un médecin spécialiste en écriture de rapports de cas médicaux. Tu es chargé d'écrire un rapport de cas médical en utilisant les entités fournies. Le rapport doit être détaillé, clair et suivre un style formel et professionnel, typique des publications médicales, des exemple te sera donné ci-dessous :

           Voici des exemple pour te guider :
           {exemples_str}

           Rédige un texte fluide et continu, sans utiliser de paragraphes distincts. Inspire toi de l'exemple ci-dessus pour adapter le style de la génération.
           Taille du cas à générer:
           {token_count}
           """
       user_prompt = f"""Démarre le texte avec les mots {start_sentence}.

       Entités Extraites pour le Cas à Générer :
       {cas_a_generer}
       """
       print(start_sentence)
       with accelerator.autocast():
           messages = [
               {"role": "user", "content": prompt + user_prompt}
           ]

           input_ids = tokenizer.apply_chat_template(
               messages,
               tokenize=True,
               add_generation_prompt=True,
               return_tensors="pt"
           ).to(accelerator.device)

           with torch.no_grad():
               outputs = model.generate(
                   input_ids,
                   max_new_tokens=2048,
                   do_sample=True,
                   temperature=0.2,
                   top_k=20,
                   top_p=0.85
               )
       del input_ids
       response = tokenizer.decode(outputs[0])
       end_time = time.time()
       duration = end_time - start_time
       avg_time.append(duration)
       print(f"Generating the sentence {j} took {duration} seconds", flush=True)
       print(response)

       new_item = {
           'model_name': model_id,
           'prompt': prompt,
           'generation': response,
           'true_text': case.get('true_text', 'N/A')
       }

       writer.writerow(new_item)

# Average duration
avg = sum(avg_time) / len(avg_time)
print(avg)
