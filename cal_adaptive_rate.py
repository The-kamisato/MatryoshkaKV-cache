import torch
import os
import numpy as np
from transformers import AutoTokenizer
from toy_experiments.piqa_gen import PIQADataset, MyPIQADataset
from toy_experiments.arc_e import ARC_E_Dataset, MyARC_E_Dataset
from toy_experiments.arc_c import ARC_C_Dataset, MyARC_C_Dataset
from toy_experiments.hellaswag import HellaSqsgDataset, MyHellaDataset
from toy_experiments.csqa import CSQA_Dataset, MyCSQA_Dataset
from toy_experiments.winogrande import WGDataset, MyWGDataset
from pca_model.modeling_pcallama_trial import PcaLlamaForCausalLM, load_from_lora_training

def load_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        data = file.read()
    data_list = list(map(float, data.split()))

    # 检查数据长度是否为 32*32
    if len(data_list) != 32 * 32:
        raise ValueError("Not 1024!!!")

    # 转换为 32x32 张量
    tensor = torch.tensor(data_list).view(32, 32).long()
    return tensor

def save_tensor_as_txt(tensor, file_name):
    """
    """
    # 将PyTorch张量转换为NumPy数组
    tensor_np = tensor.cpu().numpy()
    
    # 保存为txt文件
    np.savetxt(file_name, tensor_np, fmt='%.5f', delimiter=' ')
    print(f"result has been saved to {file_name}")

num_samples = 32
layer_num = 32
head_num = 32
interval = 32
path = "./PCA_kvcache/checkpoint/mistral-7B"

model = load_from_lora_training(
    pretrained_model_name_or_path = path,        # change when finetune on chat
    checkpoint_dir = "LLaMA-Factory/saves/LLaMA-7B/distillation/redpajama/train_only_proj_16_64_64_mistral",      # change
    lora_trained=False,
    train_key=False,
    train_value=False,
).cuda()
tokenizer = AutoTokenizer.from_pretrained(path)
# dataset = PIQADataset()
# dataset = ARC_C_Dataset()
# dataset = ARC_E_Dataset()
# dataset = HellaSqsgDataset()
# dataset = CSQA_Dataset()
# dataset = WGDataset()

dataset = MyWGDataset(tokenizer=tokenizer)    
# dataset = MyPIQADataset(tokenizer=tokenizer)
# dataset = MyHellaDataset(tokenizer=tokenizer)
# dataset = MyARC_C_Dataset(tokenizer=tokenizer)
# dataset = MyARC_E_Dataset(tokenizer=tokenizer)
# dataset = MyCSQA_Dataset(tokenizer=tokenizer)

def get_full_hidden_states(model, tokenizer, prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model_output = model(
            input_ids=inputs.input_ids.cuda(),
            key_truncate_index=torch.full((32, 32), 128, dtype=torch.long), 
            value_truncate_index=torch.full((32, 32), 128, dtype=torch.long),
            output_hidden_states=True,
        )
    return model_output.hidden_states


def get_reduced_hidden_states(model, tokenizer, prompt, key_truncate_index, value_truncate_index):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model_output = model(
            inputs.input_ids.cuda(), 
            key_truncate_index=key_truncate_index, 
            value_truncate_index=value_truncate_index, 
            output_hidden_states=True,
        )
    all_hidden_states = model_output.hidden_states
    return all_hidden_states

def compute_truncation_error(full_hidden_states, reduced_hidden_states):
    full_hidden_state = full_hidden_states[-1]
    reduced_hidden_state = reduced_hidden_states[-1]
    truncation_error = torch.norm(full_hidden_state - reduced_hidden_state) / torch.norm(full_hidden_state)
    return truncation_error

def update_result_truncate_rate(result_truncate_index, error_matrix, update_num):
    update_cnt = 0
    sorted_indices = torch.argsort(error_matrix.view(-1))
    sorted_indices_2d = [(index // error_matrix.size(1), index % error_matrix.size(1)) for index in sorted_indices]
    for index in sorted_indices_2d:
        if result_truncate_index[index] != interval:
            result_truncate_index[index] -= interval
            update_cnt += 1
        if update_cnt == update_num:
            break

    return result_truncate_index

def update_two_result_truncate_rate(
        result_key_truncate_index, 
        result_value_truncate_index, 
        key_error_matrix, 
        value_error_matrix, 
        update_num,
    ):
    update_cnt = 0
    error_matrix = torch.cat((key_error_matrix.unsqueeze(dim=0), value_error_matrix.unsqueeze(dim=0)), dim = 0)      # (2, 32, 32)
    sorted_indices = torch.argsort(error_matrix.view(-1))
    sorted_indices_3d = [
        (index // (error_matrix.size(1) * error_matrix.size(2)),
         (index // error_matrix.size(2)) % error_matrix.size(1),
         index % error_matrix.size(2))
        for index in sorted_indices
    ]
    for index in sorted_indices_3d:
        if index[0] == 0:
            if result_key_truncate_index[index[1:]] != interval:
                result_key_truncate_index[index[1:]] -= interval
                update_cnt += 1
        if index[0] == 1:
            if result_value_truncate_index[index[1:]] != interval:
                result_value_truncate_index[index[1:]] -= interval
                update_cnt += 1
        if update_cnt == update_num:
            break
    return result_key_truncate_index, result_value_truncate_index

result_key_truncate_index = torch.full((32, 32), 128, dtype=torch.long)
result_value_truncate_index = torch.full((32, 32), 128, dtype=torch.long)

prompt = " "
for data_idx in range(num_samples):
    prompt += dataset[data_idx] + " "
    print(len(prompt))

while (torch.sum(result_key_truncate_index) + torch.sum(result_value_truncate_index)) / (32 * 32 * 128 * 2) > 0.4:
    ground_truth_hidden_states = get_reduced_hidden_states(
        model=model, tokenizer=tokenizer, prompt=prompt, 
        key_truncate_index=result_key_truncate_index, 
        value_truncate_index=result_value_truncate_index,
    )
    
    key_truncate_idx = result_key_truncate_index.clone()
    value_truncate_idx = result_value_truncate_index.clone()
    # error is used to record the error of a head -= 32:
    key_error = torch.zeros(32, 32)
    value_error = torch.zeros(32, 32)

    for layer_idx in range(layer_num):
        for head_idx in range(head_num):
            
            ############ key rate #############
            key_truncate_idx[layer_idx, head_idx] -= interval
            reduced_hidden_states = get_reduced_hidden_states(
                model=model, tokenizer=tokenizer, prompt=prompt, 
                key_truncate_index=key_truncate_idx, 
                value_truncate_index=value_truncate_idx,
            )
            key_truncate_idx[layer_idx, head_idx] += interval
            key_error[layer_idx, head_idx] = compute_truncation_error(
                full_hidden_states=ground_truth_hidden_states,
                reduced_hidden_states=reduced_hidden_states,
            )   


            ############ value rate #############
            value_truncate_idx[layer_idx, head_idx] -= interval
            reduced_hidden_states = get_reduced_hidden_states(
                model=model, tokenizer=tokenizer, prompt=prompt, 
                key_truncate_index=key_truncate_idx, 
                value_truncate_index=value_truncate_idx,
            )
            value_truncate_idx[layer_idx, head_idx] += interval
            value_error[layer_idx, head_idx] = compute_truncation_error(
                full_hidden_states=ground_truth_hidden_states,
                reduced_hidden_states=reduced_hidden_states,
            )


    ############### update ###############
    # result_key_truncate_index = update_result_truncate_rate(
    #     result_truncate_index=result_key_truncate_index,
    #     error_matrix=key_error,
    #     update_num=64 * 4)

    # result_value_truncate_index = update_result_truncate_rate(
    #     result_truncate_index=result_value_truncate_index,
    #     error_matrix=value_error,
    #     update_num=64 * 4)

    result_key_truncate_index, result_value_truncate_index = update_two_result_truncate_rate(
        result_key_truncate_index, result_value_truncate_index, key_error, value_error, 64 * 24
    )
    
    cache_used = (torch.sum(result_key_truncate_index) + torch.sum(result_value_truncate_index)) / (32 * 32 * 128 * 2)
    print("truncate_index update over!")
    save_tensor_as_txt(result_key_truncate_index, "./PCA_kvcache/hella/hella24_key_{}_sample_{}_interval_{}_rate.txt".format(num_samples, interval, cache_used))
    save_tensor_as_txt(result_value_truncate_index, "./PCA_kvcache/hella/hella24_value_{}_sample_{}_interval_{}_rate.txt".format(num_samples, interval, cache_used))

print(result_key_truncate_index)
print(result_value_truncate_index)



full_hidden_states = get_full_hidden_states(
    model=model, 
    tokenizer=tokenizer, 
    prompt=prompt,
)
print("##################")
reduced_hidden_states = get_reduced_hidden_states(
    model=model, 
    tokenizer=tokenizer, 
    prompt=prompt, 
    key_truncate_index=result_key_truncate_index, 
    value_truncate_index=result_value_truncate_index,
)
truncation_error = compute_truncation_error(
    full_hidden_states=full_hidden_states,
    reduced_hidden_states=reduced_hidden_states,
)

print(truncation_error)

print("##################")
reduced_hidden_states = get_reduced_hidden_states(
    model=model, 
    tokenizer=tokenizer, 
    prompt=prompt, 
    key_truncate_index=torch.full((32, 32), 48), 
    value_truncate_index=torch.full((32, 32), 48),
)
truncation_error = compute_truncation_error(
    full_hidden_states=full_hidden_states,
    reduced_hidden_states=reduced_hidden_states,
)

print(truncation_error)
