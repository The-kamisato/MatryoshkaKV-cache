import json

# Paths to input files
input_jsonl_file = '/liymai24/sjtu/bokai/opencompass/data/siqa/train.jsonl'
input_labels_file = '/liymai24/sjtu/bokai/opencompass/data/siqa/train-labels.lst'
output_jsonl_file = 'siqa_train.jsonl'

# Read the labels
with open(input_labels_file, 'r', encoding='utf-8') as label_file:
    labels = label_file.readlines()

# Read the JSONL file and create the new JSONL content
with open(input_jsonl_file, 'r', encoding='utf-8') as jsonl_file, open(output_jsonl_file, 'w', encoding='utf-8') as outfile:
    for i, line in enumerate(jsonl_file):
        data = json.loads(line.strip())
        
        context = data['context']
        question = data['question']
        answerA = data['answerA']
        answerB = data['answerB']
        answerC = data['answerC']
        label = int(labels[i].strip())
        
        formatted_question = f"{context}\n\nQuestion: {question}\nA. {answerA}\nB. {answerB}\nC. {answerC}\nAnswer:"
        formatted_answer = "A" if label == 0 else "B" if label == 1 else "C"
        
        new_data = {
            "question": formatted_question,
            "answer": formatted_answer
        }
        
        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

print(f"Converted file saved to {output_jsonl_file}")