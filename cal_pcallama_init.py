import torch
import transformers
import pprint
from siqa import SiqaDataset
from piqa_gen import PIQADataset
from hellaswag import HellaSqsgDataset
from gsm8k import Gsm8kDataset
from obqa import ObqaDataset
from alpaca_en import AlpacaDataset
from transformers import LlamaModel, LlamaForCausalLM, AutoTokenizer
from eigenvalues_decomposition import sum_kv_states_per_head, kv_states_per_head_convariance_decomposition
from tensor_list_save import save_tensor_list
from concat_kv_cache import concat_kv_cache
from tqdm import tqdm

class CombinedDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = len(self.datasets) * min(self.lengths)
        self.truncated_lengths = [min(self.lengths)] * len(self.datasets)
        
    def __len__(self):
        return len(self.datasets) * min(self.lengths)
    
    def __getitem__(self, idx):
        for i, length in enumerate(self.truncated_lengths):
            if idx < length:
                return self.datasets[i][idx]
            idx -= length
        raise IndexError('Index out of range')
    
    
model = LlamaForCausalLM.from_pretrained("/liymai24/sjtu/bokai/PCA_kvcache/checkpoint/Llama-2-7b-hf").cuda()
tokenizer = AutoTokenizer.from_pretrained("/liymai24/sjtu/bokai/PCA_kvcache/checkpoint/Llama-2-7b-hf")
# datasets = [SiqaDataset(), PIQADataset(), HellaSqsgDataset(), ObqaDataset()]
# dataset = CombinedDataset(datasets=datasets)
dataset = AlpacaDataset(data_path='/liymai24/sjtu/bokai/LLaMA-Factory/data/alpaca_data_en_52k.json')

print(len(dataset))

total_length = 0
num_heads = 32
layer_num = 32
covariance_key_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]
covariance_value_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]
mean_key_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]
mean_value_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]

for i in tqdm(range(10000), desc="Inferencing"):
    past_key_values_list = []
    for index in range(i * 5, (i + 1) * 5):
        # context, question = dataset[index]
        # prompt = context + question
        prompt = dataset[index]['input']
        inputs = tokenizer(prompt, return_tensors="pt")
        generate_ids = model.generate(inputs.input_ids.cuda(), return_dict_in_generate=True, max_new_tokens=500)
        
        past_key_values_list.append(generate_ids.past_key_values)
        
    new_key_values_list = concat_kv_cache(past_key_values_list)
    for past_key_values in new_key_values_list:
        covariance_key_tensor, covariance_value_tensor, mean_key_tensor, mean_value_tensor = sum_kv_states_per_head(
            covariance_key_tensor, covariance_value_tensor, mean_key_tensor, mean_value_tensor, past_key_values, total_length)
        total_length += past_key_values[0][0].shape[2]
        
        for layer_idx in range(2):
            print(mean_key_tensor[layer_idx][0][20].item(), end=" ")
        for layer_idx in range(2):
            print(mean_value_tensor[layer_idx][0][20].item(), end=" ")
        print("end")
            
            
all_layers_key_states_eigenvalues_descending, all_layers_value_states_eigenvalues_descending, all_layers_key_states_eigenvectors_descending, all_layers_value_states_eigenvectors_descending = kv_states_per_head_convariance_decomposition(
    covariance_key_tensor, covariance_value_tensor, mean_key_tensor, mean_value_tensor)

# save_tensor_list(all_layers_key_states_eigenvalues_descending, "/liymai24/sjtu/bokai/PCA_kvcache/experiment_log/siqa_key_eigen_values_3.json")
# save_tensor_list(all_layers_value_states_eigenvalues_descending, "/liymai24/sjtu/bokai/PCA_kvcache/experiment_log/siqa_value_eigen_values_3.json")

torch.save(all_layers_key_states_eigenvalues_descending, 'per_head_data/alpaca/all_layers_key_states_eigenvalues_descending.pth')
torch.save(all_layers_value_states_eigenvalues_descending, 'per_head_data/alpaca/all_layers_value_states_eigenvalues_descending.pth')
torch.save(all_layers_key_states_eigenvectors_descending, 'per_head_data/alpaca/all_layers_key_states_eigenvectors_descending.pth')
torch.save(all_layers_value_states_eigenvectors_descending, 'per_head_data/alpaca/all_layers_value_states_eigenvectors_descending.pth')
torch.save(mean_key_tensor, 'per_head_data/alpaca/all_layers_key_mean.pth')
torch.save(mean_value_tensor, 'per_head_data/alpaca/all_layers_value_mean.pth')


    
