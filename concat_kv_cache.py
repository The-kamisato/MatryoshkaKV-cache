import torch
from transformers.cache_utils import DynamicCache
from typing import List, Optional, Tuple, Union

def concat_kv_cache(past_key_values_list: List[DynamicCache], max_length = 50000) -> List[DynamicCache]:
    new_key_values_list = []
    leyer_num = len(past_key_values_list[0])
    
    current_length = 0
    
    all_layers_key_states = [[] for _ in range(leyer_num)]
    all_layers_value_states = [[] for _ in range(leyer_num)]
    
    for past_key_values in past_key_values_list:
        kv_len = past_key_values[0][0].shape[2]
        if kv_len + current_length > max_length:
            new_all_layers_key_states = []
            new_all_layers_value_states = []
            residue_length = kv_len + current_length - max_length
            for layer_idx in range(leyer_num):   
                key_states = past_key_values[layer_idx][0]
                value_states = past_key_values[layer_idx][1]
                
                key_states_1 = key_states[:, :, :-residue_length, :]
                value_states_1 = value_states[:, :, :-residue_length, :]
                all_layers_key_states[layer_idx].append(key_states_1)
                all_layers_value_states[layer_idx].append(value_states_1)
                
                key_states_2 = key_states[:, :, -residue_length:, :]
                value_states_2 = value_states[:, :, -residue_length:, :]
                new_all_layers_key_states.append(key_states_2)
                new_all_layers_value_states.append(value_states_2)
                
            max_length_cache = DynamicCache()
            max_length_cache.key_cache = [torch.cat(per_layer_key_states_list, dim=2) for per_layer_key_states_list in all_layers_key_states]
            max_length_cache.value_cache = [torch.cat(per_layer_value_states_list, dim=2) for per_layer_value_states_list in all_layers_value_states]
            new_key_values_list.append(max_length_cache)
                
            current_length = residue_length
            
            for layer_idx in range(leyer_num): 
                all_layers_key_states[layer_idx] = [new_all_layers_key_states[layer_idx]]
                all_layers_value_states[layer_idx] = [new_all_layers_value_states[layer_idx]]
        else:
            for layer_idx in range(leyer_num):   
                key_states = past_key_values[layer_idx][0]
                value_states = past_key_values[layer_idx][1]
                
                all_layers_key_states[layer_idx].append(key_states)
                all_layers_value_states[layer_idx].append(value_states)
            current_length += kv_len
    
    last_cache = DynamicCache()
    last_cache.key_cache = [torch.cat(per_layer_key_states_list, dim=2) for per_layer_key_states_list in all_layers_key_states]
    last_cache.value_cache = [torch.cat(per_layer_value_states_list, dim=2) for per_layer_value_states_list in all_layers_value_states]
    new_key_values_list.append(last_cache)

    return new_key_values_list

if __name__ == "__main__":
    key_states = torch.randn(1, 32, 157, 128).cuda()
    value_states = torch.randn(1, 32, 157, 128).cuda()
    
    past_key_values = DynamicCache()
    past_key_values.key_cache = [value_states for _ in range(32)]
    past_key_values.value_cache = [value_states for _ in range(32)]
    
    past_key_values_list = [past_key_values for _ in range(3000)]
    new_key_values_list = concat_kv_cache(past_key_values_list, max_length=20000)
    print(len(new_key_values_list))

    for new_key_values in new_key_values_list:
        print(len(new_key_values))
        print(new_key_values[0][0].shape)