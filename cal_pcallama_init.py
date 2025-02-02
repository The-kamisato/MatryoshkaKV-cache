import torch
import transformers
import os
import json
import argparse

from torch.utils.data import Dataset
from transformers import LlamaModel, LlamaForCausalLM, AutoTokenizer
from eigenvalues_decomposition import sum_kv_states_per_head, kv_states_per_head_convariance_decomposition
from concat_kv_cache import concat_kv_cache
from tqdm import tqdm

def save_tensor_list(tensor_list, json_path):
    tensor_list_as_list = [[tensor.tolist() for tensor in inner_list] for inner_list in tensor_list]
    with open(json_path, "w") as json_file:
        json.dump(tensor_list_as_list, json_file, indent=2)
    print("Tensor list values saved to", json_path)

class AlpacaDataset(Dataset):
    def __init__(self, data_path):
        """
        Args:
            data (list of dict): List of dictionaries containing 'instruction', 'input', and 'output' keys.
        """
        with open(data_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        combined_input = item['instruction'] + " " + item['input'] + " " + item['output']
        return combined_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Initialize Parameter')
    parser.add_argument("--model-path", type=str, default='meta-llama/Llama-2-7b', help="model path")
    parser.add_argument("--data-path", type=str, default='LLaMA-Factory/data/alpaca_data_en_52k.json', help="data path")
    parser.add_argument("--gpu", type=str, default='1', help="GPU index to use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    alpaca_dataset = AlpacaDataset(data_path=args.data_path)

    total_length = 0
    num_heads = 32
    layer_num = 32
    covariance_key_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]
    covariance_value_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]
    mean_key_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]
    mean_value_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]

    for i in tqdm(range(len(alpaca_dataset) // 5), desc="Inferencing"):
        past_key_values_list = []
        for index in range(i * 5, (i + 1) * 5):
            # context, question = dataset[index]
            # prompt = context + question
            prompt = alpaca_dataset[index]
            inputs = tokenizer(prompt, return_tensors="pt")
            generate_ids = model.generate(inputs.input_ids.cuda(), return_dict_in_generate=True, max_new_tokens=50)
            
            past_key_values_list.append(generate_ids.past_key_values)
            
        new_key_values_list = concat_kv_cache(past_key_values_list)
        
        for past_key_values in new_key_values_list:
            covariance_key_tensor, covariance_value_tensor, mean_key_tensor, mean_value_tensor = sum_kv_states_per_head(
                covariance_key_tensor, covariance_value_tensor, mean_key_tensor, mean_value_tensor, past_key_values, total_length)
            total_length += past_key_values[0][0].shape[2]
                
    all_layers_key_states_eigenvalues_descending, all_layers_value_states_eigenvalues_descending, all_layers_key_states_eigenvectors_descending, all_layers_value_states_eigenvectors_descending = kv_states_per_head_convariance_decomposition(
        covariance_key_tensor, covariance_value_tensor, mean_key_tensor, mean_value_tensor)
    print(all_layers_key_states_eigenvalues_descending)

    if not os.path.exists("alpaca_PCA_init"):
        os.makedirs("alpaca_PCA_init")

    torch.save(all_layers_key_states_eigenvalues_descending, 'alpaca_PCA_init/all_layers_key_states_eigenvalues_descending.pth')
    torch.save(all_layers_value_states_eigenvalues_descending, 'alpaca_PCA_init/all_layers_value_states_eigenvalues_descending.pth')
    torch.save(all_layers_key_states_eigenvectors_descending, 'alpaca_PCA_init/all_layers_key_states_eigenvectors_descending.pth')
    torch.save(all_layers_value_states_eigenvectors_descending, 'alpaca_PCA_init/all_layers_value_states_eigenvectors_descending.pth')
    torch.save(mean_key_tensor, 'alpaca_PCA_init/all_layers_key_mean.pth')
    torch.save(mean_value_tensor, 'alpaca_PCA_init/all_layers_value_mean.pth')


    
