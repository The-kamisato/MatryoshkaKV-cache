import torch
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import DynamicCache
from concat_kv_cache import concat_kv_cache

def sum_kv_states(
    covariance_key_tensor: List[torch.Tensor], 
    covariance_value_tensor: List[torch.Tensor], 
    mean_key_tensor: List[torch.Tensor], 
    mean_value_tensor: List[torch.Tensor], 
    past_key_values: DynamicCache,
    all_past_key_values_length: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    key_states_all_layers = [key_states_per_layer[0] for key_states_per_layer in past_key_values]
    value_states_all_layers = [value_states_per_layer[1] for value_states_per_layer in past_key_values]
    
    for layer_idx, key_states in enumerate(key_states_all_layers):
        bsz, num_heads, kv_len, head_dim = key_states.size()          # bsz, num_heads, kv_len
        key_states = key_states.transpose(1, 2).reshape(bsz, kv_len, num_heads * head_dim)
        key_states = key_states.view(-1, num_heads * head_dim)
        
        key_states_mean = torch.mean(key_states, dim = 0)
        key_states_dot_key_states = key_states.transpose(0, 1) @ key_states
        
        mean_key_tensor[layer_idx] = mean_key_tensor[layer_idx] * (all_past_key_values_length / (all_past_key_values_length + kv_len)) + key_states_mean * (kv_len / (all_past_key_values_length + kv_len))
        covariance_key_tensor[layer_idx] = covariance_key_tensor[layer_idx] * (all_past_key_values_length / (all_past_key_values_length + kv_len)) + key_states_dot_key_states / (all_past_key_values_length + kv_len)
        
    for layer_idx, value_states in enumerate(value_states_all_layers):
        bsz, num_heads, kv_len, head_dim = value_states.size()
        value_states = value_states.transpose(1, 2).reshape(bsz, kv_len, num_heads * head_dim)
        value_states = value_states.view(-1, num_heads * head_dim)
        value_states_mean = torch.mean(value_states, dim = 0)
        value_states_dot_value_states = value_states.transpose(0, 1) @ value_states
        
        mean_value_tensor[layer_idx] = mean_value_tensor[layer_idx] * (all_past_key_values_length / (all_past_key_values_length + kv_len)) + value_states_mean * (kv_len / (all_past_key_values_length + kv_len))
        covariance_value_tensor[layer_idx] = covariance_value_tensor[layer_idx] * (all_past_key_values_length / (all_past_key_values_length + kv_len)) + value_states_dot_value_states / (all_past_key_values_length + kv_len)
    
    return covariance_key_tensor, covariance_value_tensor, mean_key_tensor, mean_value_tensor

def sum_kv_states_per_head(
    covariance_key_tensor: List[List[torch.Tensor]], 
    covariance_value_tensor: List[List[torch.Tensor]], 
    mean_key_tensor: List[List[torch.Tensor]], 
    mean_value_tensor: List[List[torch.Tensor]], 
    past_key_values: DynamicCache,
    all_past_key_values_length: int,
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
    key_states_all_layers = [key_states_per_layer[0] for key_states_per_layer in past_key_values]
    value_states_all_layers = [value_states_per_layer[1] for value_states_per_layer in past_key_values]
    
    for layer_idx, all_heads_key_states in enumerate(key_states_all_layers):
        head_num = all_heads_key_states.shape[1]
        for head_idx in range(head_num):
            key_states = all_heads_key_states[:, head_idx, :, :]
            _, kv_len, head_dim = key_states.size()
            key_states = key_states.view(-1, head_dim)
            key_states_mean = torch.mean(key_states, dim = 0)
            key_states_dot_key_states = key_states.transpose(0, 1) @ key_states
            
            mean_key_tensor[layer_idx][head_idx] = mean_key_tensor[layer_idx][head_idx] * (all_past_key_values_length / (all_past_key_values_length + kv_len)) + key_states_mean * (kv_len / (all_past_key_values_length + kv_len))
            covariance_key_tensor[layer_idx][head_idx] = covariance_key_tensor[layer_idx][head_idx] * (all_past_key_values_length / (all_past_key_values_length + kv_len)) + key_states_dot_key_states / (all_past_key_values_length + kv_len)
    
    for layer_idx, all_heads_value_states in enumerate(value_states_all_layers):
        head_num = all_heads_value_states.shape[1]
        for head_idx in range(head_num):
            value_states = all_heads_value_states[:, head_idx, :, :]
            _, kv_len, head_dim = value_states.size()
            value_states = value_states.view(-1, head_dim)
            value_states_mean = torch.mean(value_states, dim = 0)
            value_states_dot_value_states = value_states.transpose(0, 1) @ value_states
            
            mean_value_tensor[layer_idx][head_idx] = mean_value_tensor[layer_idx][head_idx] * (all_past_key_values_length / (all_past_key_values_length + kv_len)) + value_states_mean * (kv_len / (all_past_key_values_length + kv_len))
            covariance_value_tensor[layer_idx][head_idx] = covariance_value_tensor[layer_idx][head_idx] * (all_past_key_values_length / (all_past_key_values_length + kv_len)) + value_states_dot_value_states / (all_past_key_values_length + kv_len)
    
    return covariance_key_tensor, covariance_value_tensor, mean_key_tensor, mean_value_tensor


def kv_states_convariance_decomposition(
    all_layers_covariance_key_tensor: List[torch.Tensor], 
    all_layers_covariance_value_tensor: List[torch.Tensor],
    all_layers_mean_key_tensor: List[torch.Tensor], 
    all_layers_mean_value_tensor: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:  
    all_layers_key_states_eigenvalues_descending = []
    all_layers_key_states_eigenvectors_descending = []
    all_layers_value_states_eigenvalues_descending = []
    all_layers_value_states_eigenvectors_descending = []
    
    for covariance_key_tensor, mean_key_tensor in zip(all_layers_covariance_key_tensor, all_layers_mean_key_tensor):
        mean_key_tensor = mean_key_tensor.unsqueeze(dim = 0)
        covariance_key_tensor = covariance_key_tensor - mean_key_tensor.transpose(0, 1) @ mean_key_tensor
        covariance_key_tensor = covariance_key_tensor.to(torch.float32).cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_key_tensor)
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
        
        key_states_eigenvalues_descending, indices_descending = eigenvalues.flip(0), torch.arange(eigenvalues.numel()).flip(0)
        key_states_eigenvectors_descending = eigenvectors[:, indices_descending]
        all_layers_key_states_eigenvalues_descending.append(key_states_eigenvalues_descending)
        all_layers_key_states_eigenvectors_descending.append(key_states_eigenvectors_descending)
    
    for covariance_value_tensor, mean_value_tensor in zip(all_layers_covariance_value_tensor, all_layers_mean_value_tensor):
        mean_value_tensor = mean_value_tensor.unsqueeze(dim = 0)
        covariance_value_tensor = covariance_value_tensor - mean_value_tensor.transpose(0, 1) @ mean_value_tensor
        covariance_value_tensor = covariance_value_tensor.to(torch.float32).cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_value_tensor)
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
        
        value_states_eigenvalues_descending, indices_descending = eigenvalues.flip(0), torch.arange(eigenvalues.numel()).flip(0)
        value_states_eigenvectors_descending = eigenvectors[:, indices_descending]
        all_layers_value_states_eigenvalues_descending.append(value_states_eigenvalues_descending)
        all_layers_value_states_eigenvectors_descending.append(value_states_eigenvectors_descending)
    return all_layers_key_states_eigenvalues_descending, all_layers_value_states_eigenvalues_descending, all_layers_key_states_eigenvectors_descending, all_layers_value_states_eigenvectors_descending

def kv_states_per_head_convariance_decomposition(
    all_layers_covariance_key_tensor: List[List[torch.Tensor]], 
    all_layers_covariance_value_tensor: List[List[torch.Tensor]],
    all_layers_mean_key_tensor: List[List[torch.Tensor]], 
    all_layers_mean_value_tensor: List[List[torch.Tensor]], 
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], List[List[torch.Tensor]], List[List[torch.Tensor]]]:  
    all_layers_key_states_eigenvalues_descending = []
    all_layers_key_states_eigenvectors_descending = []
    all_layers_value_states_eigenvalues_descending = []
    all_layers_value_states_eigenvectors_descending = []
    
    for all_heads_covariance_key_tensor, all_heads_mean_key_tensor in zip(all_layers_covariance_key_tensor, all_layers_mean_key_tensor):
        all_heads_key_states_eigenvalues_descending = []
        all_heads_key_states_eigenvectors_descending = []
        for covariance_key_tensor, mean_key_tensor in zip(all_heads_covariance_key_tensor, all_heads_mean_key_tensor):
            mean_key_tensor = mean_key_tensor.unsqueeze(dim = 0)
            covariance_key_tensor = covariance_key_tensor - mean_key_tensor.transpose(0, 1) @ mean_key_tensor
            covariance_key_tensor = covariance_key_tensor.to(torch.float32).cpu()
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_key_tensor)
            eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
            
            indices = torch.argsort(eigenvalues, descending=True)
            # print(indices)
            key_states_eigenvalues_descending = eigenvalues[indices]
            key_states_eigenvectors_descending = eigenvectors[:, indices]
            
            all_heads_key_states_eigenvalues_descending.append(key_states_eigenvalues_descending)
            all_heads_key_states_eigenvectors_descending.append(key_states_eigenvectors_descending)
            
        all_layers_key_states_eigenvalues_descending.append(all_heads_key_states_eigenvalues_descending)
        all_layers_key_states_eigenvectors_descending.append(all_heads_key_states_eigenvectors_descending)
    
    for all_heads_covariance_value_tensor, all_heads_mean_value_tensor in zip(all_layers_covariance_value_tensor, all_layers_mean_value_tensor):
        all_heads_value_states_eigenvalues_descending = []
        all_heads_value_states_eigenvectors_descending = []
        for covariance_value_tensor, mean_value_tensor in zip(all_heads_covariance_value_tensor, all_heads_mean_value_tensor):
            mean_value_tensor = mean_value_tensor.unsqueeze(dim = 0)
            covariance_value_tensor = covariance_value_tensor - mean_value_tensor.transpose(0, 1) @ mean_value_tensor
            covariance_value_tensor = covariance_value_tensor.to(torch.float32).cpu()
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_value_tensor)
            eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
            
            # value_states_eigenvalues_descending, indices_descending = eigenvalues.flip(0), torch.arange(eigenvalues.numel()).flip(0)
            # value_states_eigenvectors_descending = eigenvectors[:, indices_descending]
            
            indices = torch.argsort(eigenvalues, descending=True)
            # print(indices)
            value_states_eigenvalues_descending = eigenvalues[indices]
            value_states_eigenvectors_descending = eigenvectors[:, indices]
            
            all_heads_value_states_eigenvalues_descending.append(value_states_eigenvalues_descending)
            all_heads_value_states_eigenvectors_descending.append(value_states_eigenvectors_descending)
        
        all_layers_value_states_eigenvalues_descending.append(all_heads_value_states_eigenvalues_descending)
        all_layers_value_states_eigenvectors_descending.append(all_heads_value_states_eigenvectors_descending)
    return all_layers_key_states_eigenvalues_descending, all_layers_value_states_eigenvalues_descending, all_layers_key_states_eigenvectors_descending, all_layers_value_states_eigenvectors_descending

if __name__ == "__main__":
    total_length = 0
    num_heads = 32
    layer_num = 32
    covariance_key_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]
    covariance_value_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]
    mean_key_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]
    mean_value_tensor = [[torch.tensor(0) for _ in range(num_heads)] for _ in range(layer_num)]
    
    key_states = torch.randn(1, 32, 157, 128).cuda()
    value_states = torch.randn(1, 32, 157, 128).cuda()
    
    past_key_values = DynamicCache()
    past_key_values.key_cache = [key_states for _ in range(32)]
    past_key_values.value_cache = [value_states for _ in range(32)]
    
    past_key_values_list = [past_key_values for _ in range(200)]
    new_key_values_list = concat_kv_cache(past_key_values_list, max_length=20000)
    print(len(new_key_values_list))
    for past_key_values in new_key_values_list:
        covariance_key_tensor, covariance_value_tensor, mean_key_tensor, mean_value_tensor = sum_kv_states_per_head(
            covariance_key_tensor, covariance_value_tensor, mean_key_tensor, mean_value_tensor, past_key_values, total_length)
        total_length += past_key_values[0][0].shape[2]
        
    all_layers_key_states_eigenvalues_descending, all_layers_value_states_eigenvalues_descending, _, _ = kv_states_per_head_convariance_decomposition(
        covariance_key_tensor, covariance_value_tensor, mean_key_tensor, mean_value_tensor)
    
    print(len(all_layers_key_states_eigenvalues_descending))
    print(len(all_layers_key_states_eigenvalues_descending[0]))
    