import json

# Paths to input files
input_jsonl_file = '/liymai24/sjtu/bokai/opencompass/data/hellaswag/hellaswag_train.jsonl'
output_jsonl_file = 'hellaswag_train.jsonl'



# Read the JSONL file and create the new JSONL content
with open(input_jsonl_file, 'r', encoding='utf-8') as jsonl_file, open(output_jsonl_file, 'w', encoding='utf-8') as outfile:
    for line in jsonl_file:
        data = json.loads(line.strip())
        qa_data = {
            "context": data["ctx"],
            "question": "What will happen next?",
            "options": data["endings"],
            "answer": data["endings"][data["label"]]
        }
        
        options = qa_data["options"]
        formatted_question = f"{qa_data['context']}\n\nQuestion: {qa_data['question']}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer:"
        formatted_answer = "A" if data["label"] == 0 else "B" if data["label"] == 1 else "C" if data["label"] == 2 else "D"
        
        new_data = {
            "question": formatted_question,
            "answer": formatted_answer
        }
        
        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

print(f"Converted file saved to {output_jsonl_file}")