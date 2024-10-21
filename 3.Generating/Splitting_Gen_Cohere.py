import csv

def split_and_keep_second_part(text, word):
    parts = text.split(word, 1)
    return parts[1] if len(parts) > 1 else ""

def split_and_keep_first_part(text, word):
    text = text.replace('\n', '')
    parts = text.split(word, 1)
    first_part = parts[0].strip() if len(parts) > 0 else text.strip()
    
    return first_part

with open('command_r_plus_4bit_one_shot_gen.csv', newline='', encoding='utf-8') as f:
    cases = list(csv.DictReader(f))

mot_division = "<|CHATBOT_TOKEN|>"
mot_division_2 = "<|END_OF_TURN_TOKEN|>"

processed_cases = []
for case in cases:
    case["generation"] = split_and_keep_second_part(case["generation"], mot_division)
    case["generation"] = split_and_keep_first_part(case["generation"], mot_division_2)
    processed_cases.append(case)

with open('command_r_plus_4bit_one_shot_gen_final.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=cases[0].keys())
    writer.writeheader()
    writer.writerows(processed_cases)
