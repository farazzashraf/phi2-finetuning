import json

with open('data/coco.en', 'r', encoding='utf-8') as en_file, open('data/coco.ml', 'r', encoding='utf-8') as ml_file:
    en_lines = en_file.readlines()
    ml_lines = ml_file.readlines()
    
assert len(en_lines) == len(ml_lines), "The number of lines in the files do not match."

with open('coco_dataset_bi.jsonl', 'w', encoding='utf-8') as jsonl_file:
    for en, ml in zip(en_lines, ml_lines):
        en = en.strip()
        ml = ml.strip()
        if en and ml:
            jsonl_line_en_ml = {
                "instruction": "Translate the sentence to Malayalam.",
                "input": en,
                "output": ml
            }
            
        # Write the dictionary as a JSON object to the file
        jsonl_file.write(json.dumps(jsonl_line_en_ml, ensure_ascii=False) + '\n')
        
        jsonl_line_ml_en = {
            "instruction": "Translate the sentence to English.",
            "input": ml,
            "output": en
        }
        jsonl_file.write(json.dumps(jsonl_line_ml_en, ensure_ascii=False) + '\n')    
