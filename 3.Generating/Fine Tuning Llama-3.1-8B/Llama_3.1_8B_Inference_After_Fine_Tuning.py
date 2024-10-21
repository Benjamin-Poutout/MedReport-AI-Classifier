import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from huggingface_hub import login
from accelerate import Accelerator
import csv
import re  # Import regex module

accelerator = Accelerator()

# Load the fine-tuned model
login(token="hugging_face_token_here")
model_name = "llama_3.1/checkpoint-6820"  # Replace with your fine-tuned model path
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load the modified test data
test_file = "llama_3_finetune_data_test.jsonl"

# Inference function
def generate_response(instruction, input_text, max_length=3000):
    # Create prompt with ### Instruction and ### Input
    prompt = f"### Instruction :\n{instruction}\n\n### Input :\n{input_text}\n\n### Response :"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the response using the model
    output = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_p=0.95,
        temperature=0.4
    )
    
    # Decode the generated response
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the generated text after ### Response
    response_start = generated_text.find("### Response :")
    if response_start != -1:
        return generated_text[response_start + len("### Response :"):].strip()
    
    return generated_text

# Read the test data and perform inference for each line
with open(test_file, "r", encoding="utf-8") as f, open('llama_3.1_fine_tune.csv', "w", newline='', encoding="utf-8") as out_f:
    csv_writer = csv.writer(out_f)
    
    # Write the CSV header
    csv_writer.writerow(["instruction", "input", "generated_response"])
    
    for line in f:
        print(f"Processing line: {line.strip()}")
        entry = json.loads(line)
        
        # Extract the Instruction and Input parts from each text
        text = entry['text'].strip()  # Clean the text
        
        # Use regex to find the tags
        instruction_match = re.search(r"###\s*Instruction\s*:\s*(.*?)(?=###\s*Input\s*:)", text, re.DOTALL)
        input_match = re.search(r"###\s*Input\s*:\s*(.*?)(?=###\s*Response\s*:)", text, re.DOTALL)

        # Get the values if the tags are found
        instruction = instruction_match.group(1).strip() if instruction_match else None
        input_text = input_match.group(1).strip() if input_match else None
        
        if instruction and input_text:
            # Generate the response
            response = generate_response(instruction, input_text)
            
            # Write the result to CSV format
            csv_writer.writerow([instruction, input_text, response])
            
            # Optionally print results to console
            print(f"Instruction: {instruction}")
            print(f"Input: {input_text}")
            print(f"Generated Response: {response}")
            print("-" * 80)
        else:
            print("Instruction or Input not found in the line.")

