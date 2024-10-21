import math
import os
import torch
import torch.nn.functional as F
import warnings
import transformers
import random
import scipy
import numpy as np
from functools import partial
from transformers import AutoConfig
from .PcaCache_new_ver import PcaDynamicCache
from typing import List, Optional, Tuple, Union
from transformers.models.llama.modeling_llama import *
from transformers.cache_utils import Cache, DynamicCache

_CONFIG_FOR_DOC = "LlamaConfig"

from typing import List, Optional, Tuple
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import PeftModel
import wandb
wandb.login()
wandb.init(project="lora_pcallama")

original_all_layers_mean_key_states = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/piqa/all_layers_key_mean.pth')
original_all_layers_mean_value_states = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/piqa/all_layers_value_mean.pth')
original_all_layers_key_states_eigenvectors_descending = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/piqa/all_layers_key_states_eigenvectors_descending.pth')
original_all_layers_value_states_eigenvectors_descending = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/piqa/all_layers_value_states_eigenvectors_descending.pth')

# original_all_layers_mean_key_states = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/gsm8k/all_layers_key_mean.pth')
# original_all_layers_mean_value_states = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/gsm8k/all_layers_value_mean.pth')
# original_all_layers_key_states_eigenvectors_descending = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/gsm8k/all_layers_key_states_eigenvectors_descending.pth')
# original_all_layers_value_states_eigenvectors_descending = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/gsm8k/all_layers_value_states_eigenvectors_descending.pth')

# original_all_layers_mean_key_states = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/alpaca/all_layers_key_mean.pth')
# original_all_layers_mean_value_states = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/alpaca/all_layers_value_mean.pth')
# original_all_layers_key_states_eigenvectors_descending = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/alpaca/all_layers_key_states_eigenvectors_descending.pth')
# original_all_layers_value_states_eigenvectors_descending = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/alpaca/all_layers_value_states_eigenvectors_descending.pth')

# original_all_layers_mean_key_states = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/mixing/all_layers_key_mean.pth')
# original_all_layers_mean_value_states = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/mixing/all_layers_value_mean.pth')
# original_all_layers_key_states_eigenvectors_descending = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/mixing/all_layers_key_states_eigenvectors_descending.pth')
# original_all_layers_value_states_eigenvectors_descending = torch.load('/liymai24/sjtu/bokai/LLaMA-Factory/src/llamafactory/model/custom_model/per_head_data/mixing/all_layers_value_states_eigenvectors_descending.pth')

def change_label_piqa(input_ids, label):
    end_index = ((label == 350) | (label == 319)).nonzero(as_tuple=True)[0].item()
    updated_label = label.clone()
    updated_label[:end_index] = torch.where(updated_label[:end_index] == -100, input_ids[1:end_index+1], updated_label[:end_index])

    return updated_label

def load_from_lora_training(
    pretrained_model_name_or_path, 
    checkpoint_dir, 
    device='cuda', 
    *model_args, 
    **kwargs
):
    config = kwargs.pop("config", None)
    if config is None:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

    head_dim = config.hidden_size // config.num_attention_heads
    unitary_transform = torch.load(os.path.join(checkpoint_dir, 'unitary_transform_weight.bin'))
    all_layers_mean_key_states = [torch.tensor(0) for _ in range(config.num_hidden_layers)]
    all_layers_mean_value_states = [torch.tensor(0) for _ in range(config.num_hidden_layers)]
    all_layers_key_states_eigenvectors_descending = [torch.tensor(0) for _ in range(config.num_hidden_layers)]
    all_layers_value_states_eigenvectors_descending = [torch.tensor(0) for _ in range(config.num_hidden_layers)]
    for name, param in unitary_transform.items():
        layer_idx = int(name.split('.')[4])
        if str(name.split('.')[-1]) == "mean_key_weights":
            split_mean_key = torch.split(param, 1, dim=1)             # param.shape = [1, 32, 1, 128]  
            all_layers_mean_key_states[layer_idx] = [mean_key.view(head_dim).cpu().detach().clone() for mean_key in split_mean_key]
            
        if str(name.split('.')[-1]) == "mean_value_weights":
            split_mean_value = torch.split(param, 1, dim=1)             # param.shape = [1, 32, 1, 128]  
            all_layers_mean_value_states[layer_idx] = [mean_value.view(head_dim).cpu().detach().clone() for mean_value in split_mean_value]
        
        if str(name.split('.')[-1]) == "key_unitary_transform_weights":
            split_key_unitary_transform = torch.split(param, 1, dim=0)   # param.shape = [32, 128, 128]  
            all_layers_key_states_eigenvectors_descending[layer_idx] = [key_unitary_transform.squeeze(dim = 0).cpu().detach().clone() for key_unitary_transform in split_key_unitary_transform]
            
        if str(name.split('.')[-1]) == "value_unitary_transform_weights":
            split_value_unitary_transform = torch.split(param, 1, dim=0)   # param.shape = [32, 128, 128]
            all_layers_value_states_eigenvectors_descending[layer_idx] = [value_unitary_transform.squeeze(dim = 0).cpu().detach().clone() for value_unitary_transform in split_value_unitary_transform]

    model = PcaLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        all_layers_mean_key_states = all_layers_mean_key_states, 
        all_layers_mean_value_states = all_layers_mean_value_states,
        all_layers_key_states_eigenvectors_descending = all_layers_key_states_eigenvectors_descending,
        all_layers_value_states_eigenvectors_descending = all_layers_value_states_eigenvectors_descending,
        config=config,
        *model_args,
        **kwargs
    ).to(device)
    
    # model = PcaLlamaForCausalLM.from_pretrained(
    #     pretrained_model_name_or_path,
    #     all_layers_mean_key_states = original_all_layers_mean_key_states, 
    #     all_layers_mean_value_states = original_all_layers_mean_value_states,
    #     all_layers_key_states_eigenvectors_descending = original_all_layers_key_states_eigenvectors_descending,
    #     all_layers_value_states_eigenvectors_descending = original_all_layers_value_states_eigenvectors_descending,
    #     config=config,
    #     *model_args,
    #     **kwargs
    # ).to(device)
    
    print("load PeftModel")
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')

    # 返回自定义模型实例
    return model
def merge_base_model_outputs_with_past(outputs: List[BaseModelOutputWithPast]) -> BaseModelOutputWithPast:
    """
    Merge a list of BaseModelOutputWithPast objects along the batch dimension.

    Args:
        outputs (List[BaseModelOutputWithPast]): List of BaseModelOutputWithPast objects to merge.

    Returns:
        BaseModelOutputWithPast: A single BaseModelOutputWithPast object with merged tensors.
    """
    last_hidden_states = torch.cat([output.last_hidden_state for output in outputs], dim=0) if outputs[0].last_hidden_state is not None else None
    
    past_key_values = None
    if outputs[0].past_key_values is not None:
        # Assuming all outputs have the same number of layers and past_key_values structure
        past_key_values = tuple(
            tuple(torch.cat([output.past_key_values[layer][i] for output in outputs], dim=0) for i in range(len(outputs[0].past_key_values[0])))
            for layer in range(len(outputs[0].past_key_values))
        )

    hidden_states = None
    if outputs[0].hidden_states is not None:
        hidden_states = tuple(torch.cat([output.hidden_states[i] for output in outputs], dim=0) for i in range(len(outputs[0].hidden_states)))

    attentions = None
    if outputs[0].attentions is not None:
        attentions = tuple(torch.cat([output.attentions[i] for output in outputs], dim=0) for i in range(len(outputs[0].attentions)))

    return BaseModelOutputWithPast(
        last_hidden_state=last_hidden_states,
        past_key_values=past_key_values,
        hidden_states=hidden_states,
        attentions=attentions
    )

# Example usage
# outputs = [output1, output2, ...]
# merged_output = merge_base_model_outputs_with_past(outputs)


def check_tensor(tensor, step_name):
    if not torch.is_tensor(tensor):
        return

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    inf_mask = torch.isinf(tensor)
    inf_indices = torch.nonzero(inf_mask)

    # has_neg = (tensor < 0).any().item()

    if has_nan or has_inf:
        print(f"Step {step_name}:")
        if has_nan:
            print("  Contains NaN")
        if has_inf:
            print("  Contains Inf", inf_indices)
        # if has_neg:
        #     print("  Contains negative values")
            
def unitary_transform(attention_states: torch.Tensor, unitary_transform_matrix: torch.Tensor, mean_states: torch.Tensor = torch.tensor(0), truncate_index: int = 128):
    transformed_attention_states = torch.einsum('bhld,hdc->bhlc', (attention_states - mean_states), unitary_transform_matrix[..., :truncate_index])
    return transformed_attention_states
    
class PcaLlamaAttention(LlamaAttention):
    def __init__(
        self, 
        config: LlamaConfig, 
        mean_key_states: List[torch.Tensor],
        mean_value_states: List[torch.Tensor],
        key_unitary_transform_matrix: List[torch.Tensor], 
        value_unitary_transform_matrix: List[torch.Tensor], 
        train_key: bool = True, 
        train_value: bool = True, 
        layer_idx: Optional[int] = None
    ):
        super().__init__(config, layer_idx)
        self.config = config
        self.train_key = train_key
        self.train_value = train_value    
        
        # len(mean_key_states) = head_num = 32 -> (1, head_num, 1, 128)
        
        # mean_key_states_tensor = torch.stack(mean_key_states, dim=0).unsqueeze(dim=0).unsqueeze(dim=2).to(self.dtype)
        # mean_value_states_tensor = torch.stack(mean_value_states, dim=0).unsqueeze(dim=0).unsqueeze(dim=2).to(self.dtype)
        
        ones = torch.ones(1, 32, 1, 128)
        mean_key_states_tensor = ones * torch.stack(mean_key_states, dim=0).unsqueeze(dim=0).unsqueeze(dim=2).to(ones.dtype).to(ones.device)
        mean_value_states_tensor = ones * torch.stack(mean_value_states, dim=0).unsqueeze(dim=0).unsqueeze(dim=2).to(ones.dtype).to(ones.device)
        
        # len(self.unitary_transform_matrix) = head_num = 32 -> (head_num, 128, 128)
        # trainable weights:
        
        self.mean_key_weights = nn.Parameter(mean_key_states_tensor) if train_key else None
        self.mean_value_weights = nn.Parameter(mean_value_states_tensor) if train_value else None
        # self.mean_key_weights = nn.Parameter(torch.zeros(1, 32, 1, 128)) if train_key else None
        # self.mean_value_weights = nn.Parameter(torch.zeros(1, 32, 1, 128)) if train_value else None
        
        # [32, 128, 128]
        
        key_unitary_transform_matrix_tensor = torch.stack(key_unitary_transform_matrix, dim=0)
        value_unitary_transform_matrix_tensor = torch.stack(value_unitary_transform_matrix, dim=0)
        print(key_unitary_transform_matrix_tensor.shape)
        self.key_unitary_transform_weights = nn.Parameter(key_unitary_transform_matrix_tensor) if train_key else None       
        self.value_unitary_transform_weights = nn.Parameter(value_unitary_transform_matrix_tensor) if train_value else None
        # self.key_unitary_transform_weights = nn.Parameter(self.orthogonal_(torch.randn(32, 128, 128))) if train_key else None
        # self.value_unitary_transform_weights = nn.Parameter(self.orthogonal_(torch.randn(32, 128, 128))) if train_value else None
      
        self.merged_weights = False
        
        self.register_buffer('mean_key_states', mean_key_states_tensor)
        self.register_buffer('mean_value_states', mean_value_states_tensor)
        self.register_buffer('key_unitary_transform_matrix', key_unitary_transform_matrix_tensor)
        self.register_buffer('value_unitary_transform_matrix', value_unitary_transform_matrix_tensor)
    
    # cayley map between training weights and orthogonal matrix
    def orthogonal_(self, X, orthogonal_type='cayley'):
        X = X.to(torch.float32)
        if orthogonal_type=='cayley' or orthogonal_type=='exp':
            A = X - X.mH
            if orthogonal_type=='cayley':
                n = X.shape[-1]
                Id = torch.eye(n, dtype=X.dtype, device=X.device)
                Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
            else:
                Q = torch.matrix_exp(A)
        elif orthogonal_type=='householder':
            A = X.tril(diagonal=-1)
            tau = 2. / (1. + (A * A).sum(dim=-2))
            Q = torch.linalg.householder_product(A, tau)
            Q = Q * X.diagonal(dim1=-2, dim2=-1).int().unsqueeze(-2)
        else:
            raise ValueError(
                "orthogonal_type should be cayley or exp or householder"
            )
        return Q.to(self.dtype)
    
    # initialize the weight of output_proj
    def merge_wights(self):
        o_proj_matrix = self.o_proj.weight.data.t()
        weight = torch.einsum('hdc,hcb->hdb', self.value_unitary_transform_matrix.transpose(1, 2), o_proj_matrix.reshape(self.num_heads, self.head_dim, self.hidden_size))
        bias = torch.matmul(self.mean_value_states.view(self.config.hidden_size), o_proj_matrix)
        self.register_buffer('transformed_weight', weight)
        self.register_buffer('transformed_bias', bias)
        
        self.merged_weights = True
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype 
    
      
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        key_truncate_index: int = 128,
        value_truncate_index: int = 128, 
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        mean_key_states = self.mean_key_weights if self.train_key else self.mean_key_states
        mean_value_states = self.mean_value_weights if self.train_value else self.mean_value_states
        key_unitary_transform_matrix = self.key_unitary_transform_weights if self.train_key else self.key_unitary_transform_matrix
        value_unitary_transform_matrix = self.value_unitary_transform_weights if self.train_value else self.value_unitary_transform_matrix
        
        check_tensor(key_unitary_transform_matrix, 'key')
        check_tensor(value_unitary_transform_matrix, 'value')
        # if self.training and self.merged_weights and self.train_value:             # when training, we need to update the weights of output_proj from time to time
        #     self.update_o_proj_weights(mean_value_states, value_unitary_transform_matrix)
            
        # if not self.merged_weights and not self.training:            
        #     self.merge_wights()
            
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # value_states1 = value_states
        
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        original_query_states = query_states
        
        query_states = unitary_transform(attention_states=query_states, unitary_transform_matrix=key_unitary_transform_matrix, truncate_index=key_truncate_index)
        key_states = unitary_transform(attention_states=key_states, unitary_transform_matrix=key_unitary_transform_matrix, mean_states=mean_key_states, truncate_index=key_truncate_index)
        value_states = unitary_transform(attention_states=value_states, unitary_transform_matrix=value_unitary_transform_matrix, mean_states=mean_value_states, truncate_index=value_truncate_index)
                
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(                               # key: (bsz, num_heads, kv_len, truncated_dim < 128)   value: (bsz, num_heads, kv_len, truncated_dim = 128)
                key_states, value_states, self.layer_idx, cache_kwargs, 
                truncate_key = True, truncate_value = True, 
                key_truncate_index = key_truncate_index, value_truncate_index = value_truncate_index,
            )
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) + torch.matmul(original_query_states, mean_key_states.transpose(2, 3))) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = torch.einsum('bhld,hdc->bhlc', attn_output, value_unitary_transform_matrix[..., :value_truncate_index].transpose(1, 2)) + mean_value_states
        attn_output = attn_output.transpose(1, 2).contiguous()               # (bsz, q_len, self.num_heads, self.head_dim < 128)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        # # custom output_proj
        # if self.training:
        #     attn_output = torch.einsum('bhld,hdc->bhlc', attn_output, value_unitary_transform_matrix[..., :value_truncate_index].transpose(1, 2)) + mean_value_states
        #     attn_output = attn_output.transpose(1, 2).contiguous()               # (bsz, q_len, self.num_heads, self.head_dim < 128)
        #     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        #     attn_output = self.o_proj(attn_output)
        #     # o_proj_matrix = self.o_proj.weight.data.t()
        #     # new_weight = torch.einsum(
        #     #     'hdc,hcb->hdb', value_unitary_transform_matrix[..., :value_truncate_index].transpose(1, 2), o_proj_matrix.reshape(self.num_heads, self.head_dim, self.hidden_size)
        #     # ).reshape(-1, self.hidden_size)
        #     # new_bias = torch.matmul(mean_value_states.view(self.config.hidden_size), o_proj_matrix)
            
        #     # attn_output = torch.matmul(attn_output, new_weight) + new_bias
        # else:
        #     attn_output = attn_output.transpose(1, 2).contiguous()               # (bsz, q_len, self.num_heads, self.head_dim < 128)
        #     attn_output = attn_output.reshape(bsz, q_len, -1)
        #     attn_output = torch.matmul(attn_output, self.transformed_weight[:, :value_truncate_index, :].reshape(-1, self.hidden_size)) + self.transformed_bias 
                   
        if not output_attentions:
            attn_weights = None
            
        return attn_output, attn_weights, past_key_value

class PcaLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self, 
        config: LlamaConfig, 
        mean_key_states: List[torch.Tensor],
        mean_value_states: List[torch.Tensor],
        key_unitary_transform_matrix: List[torch.Tensor], 
        value_unitary_transform_matrix: List[torch.Tensor], 
        train_key: bool = True, 
        train_value: bool = True, 
        layer_idx: Optional[int] = None
    ):
        super().__init__(config, layer_idx)
        self.self_attn = PcaLlamaAttention(
            config=config, 
            mean_key_states=mean_key_states,
            mean_value_states=mean_value_states,
            key_unitary_transform_matrix=key_unitary_transform_matrix, 
            value_unitary_transform_matrix=value_unitary_transform_matrix,
            train_key=train_key, 
            train_value=train_value, 
            layer_idx=layer_idx,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        key_truncate_index: int = 128,
        value_truncate_index: int = 128, 
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            key_truncate_index=key_truncate_index,
            value_truncate_index=value_truncate_index, 
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class PcaLlamaModel(LlamaModel):
    def __init__(
        self, 
        config: LlamaConfig, 
        all_layers_mean_key_states: List[List[torch.Tensor]],                   
        all_layers_mean_value_states: List[List[torch.Tensor]],                   
        all_layers_key_unitary_transform_matrix: List[List[torch.Tensor]], 
        all_layers_value_unitary_transform_matrix: List[List[torch.Tensor]], 
        train_key: bool = True, 
        train_value: bool = True, 
    ):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [PcaLlamaDecoderLayer(
                config, 
                mean_key_states=all_layers_mean_key_states[layer_idx],
                mean_value_states=all_layers_mean_value_states[layer_idx],
                key_unitary_transform_matrix=all_layers_key_unitary_transform_matrix[layer_idx], 
                value_unitary_transform_matrix=all_layers_value_unitary_transform_matrix[layer_idx], 
                train_key=train_key, 
                train_value=train_value, 
                layer_idx=layer_idx,
            ) for layer_idx in range(config.num_hidden_layers)]
        )
                
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        key_truncate_index: int = 128,
        value_truncate_index: int = 128, 
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = PcaDynamicCache.from_legacy_cache(head_dim = self.config.hidden_size // self.config.num_attention_heads, past_key_values = past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    key_truncate_index,
                    value_truncate_index, 
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    key_truncate_index=key_truncate_index,
                    value_truncate_index=value_truncate_index, 
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class PcaLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        
        self.all_layers_mean_key_states = config.all_layers_mean_key_states
        self.all_layers_mean_value_states = config.all_layers_mean_value_states
        self.all_layers_key_unitary_transform_matrix = config.all_layers_key_unitary_transform_matrix
        self.all_layers_value_unitary_transform_matrix = config.all_layers_value_unitary_transform_matrix
        self.model = PcaLlamaModel( 
            config, 
            config.all_layers_mean_key_states,
            config.all_layers_mean_value_states,
            config.all_layers_key_unitary_transform_matrix, 
            config.all_layers_value_unitary_transform_matrix, 
        )
        
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        all_layers_mean_key_states = None, 
        all_layers_mean_value_states = None,
        all_layers_key_states_eigenvectors_descending = None,
        all_layers_value_states_eigenvectors_descending = None,
        *model_args, 
        **kwargs):

        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        config.all_layers_mean_key_states = all_layers_mean_key_states if all_layers_mean_key_states is not None else original_all_layers_mean_key_states  
        config.all_layers_mean_value_states = all_layers_mean_value_states if all_layers_mean_value_states is not None else original_all_layers_mean_value_states
        config.all_layers_key_unitary_transform_matrix = all_layers_key_states_eigenvectors_descending if all_layers_key_states_eigenvectors_descending is not None else original_all_layers_key_states_eigenvectors_descending
        config.all_layers_value_unitary_transform_matrix = all_layers_value_states_eigenvectors_descending if all_layers_value_states_eigenvectors_descending is not None else original_all_layers_value_states_eigenvectors_descending
        # 使用父类的方法加载模型
        model = super(PcaLlamaForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            *model_args,
            **kwargs
        )

        # 返回自定义模型实例
        return model
        
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        key_truncate_index: int = 128,
        value_truncate_index: int = 128, 
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        if self.training:
            key_truncate_index = 64           # change this
            value_truncate_index = 128            # change this
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                key_truncate_index=key_truncate_index,
                value_truncate_index=value_truncate_index,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            # outputs_list = []
            # key_truncate_index = 64
            # for value_truncate_index in [4, 8, 16, 32, 64, 128]:            # we need to train value more, so we only sample key 
            #     # key_truncate_index = random.choice([4, 8, 16, 32, 64, 128])
            #     outputs = self.model(
            #         input_ids=input_ids,
            #         attention_mask=attention_mask,
            #         position_ids=position_ids,
            #         past_key_values=past_key_values,
            #         key_truncate_index=key_truncate_index,
            #         value_truncate_index=value_truncate_index,
            #         inputs_embeds=inputs_embeds,
            #         use_cache=use_cache,
            #         output_attentions=output_attentions,
            #         output_hidden_states=output_hidden_states,
            #         return_dict=return_dict,
            #         cache_position=cache_position,
            #     )
            #     outputs_list.append(outputs)
            # outputs = merge_base_model_outputs_with_past(outputs_list)
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                key_truncate_index=key_truncate_index,
                value_truncate_index=value_truncate_index,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            length = labels.shape[1]
            labels = labels.repeat(logits.shape[0] // labels.shape[0], 1)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if self.training:
            #     for batch_idx in range(shift_labels.shape[0]):
            #         shift_labels[batch_idx] = change_label_piqa(input_ids=input_ids[batch_idx], label=shift_labels[batch_idx])

            length = length - 1
            
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            
            # print(shift_logits.shape, shift_labels.shape)       # torch.Size([3444, 32000]) torch.Size([574 * 6])
            loss = loss_fct(shift_logits, shift_labels)
            wandb.log({
                "trauncate value 64": loss_fct(shift_logits[:length], shift_labels[:length]),
                # "trauncate value 8": loss_fct(shift_logits[length: 2*length], shift_labels[length: 2*length]),
                # "trauncate value 16": loss_fct(shift_logits[2*length: 3*length], shift_labels[2*length: 3*length]),
                # "trauncate value 32": loss_fct(shift_logits[3*length: 4*length], shift_labels[3*length: 4*length]),
                # "trauncate value 64": loss_fct(shift_logits[4*length: 5*length], shift_labels[4*length: 5*length]),      
                # "trauncate value 128": loss_fct(shift_logits[5*length: 6*length], shift_labels[5*length: 6*length]), 
            })
          
            # print("trauncate value 4", loss_fct(shift_logits[:length], shift_labels[:length]))
            # print("trauncate value 8", loss_fct(shift_logits[length: 2*length], shift_labels[length: 2*length]))
            # print("trauncate value 16", loss_fct(shift_logits[2*length: 3*length], shift_labels[2*length: 3*length]))
            # print("trauncate value 32", loss_fct(shift_logits[3*length: 4*length], shift_labels[3*length: 4*length]))
            # print("trauncate value 64", loss_fct(shift_logits[4*length: 5*length], shift_labels[4*length: 5*length]))
            # print("trauncate value 128", loss_fct(shift_logits[5*length: 6*length], shift_labels[5*length: 6*length]))
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
def augment_llama_with_unitary_transform():
    transformers.models.llama.modeling_llama.LlamaForCausalLM = PcaLlamaForCausalLM
