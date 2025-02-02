import math
import os
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import warnings
import transformers
import random
from functools import partial
from transformers import AutoConfig
from .PcaCache_trial import PcaDynamicCache
from typing import List, Optional, Tuple, Union
from transformers.models.llama.modeling_llama import *
from transformers.cache_utils import Cache, DynamicCache

_CONFIG_FOR_DOC = "LlamaConfig"

from typing import List, Optional, Tuple
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import PeftModel

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

cal_key_truncate_index = torch.full((32, 32), 64, dtype = torch.long)
cal_value_truncate_index = torch.full((32, 32), 64, dtype = torch.long)

def load_from_lora_training(
        pretrained_model_name_or_path, 
        checkpoint_dir, 
        lora_trained=True, 
        ofted=True,
        train_key=True, 
        train_value=True, 
        device='cuda', 
        key_truncate_index=None,
        value_truncate_index=None,
        *model_args, 
        **kwargs
    ):
    config = kwargs.pop("config", None)
    if config is None:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    config.rope_scaling = None
    config.pretraining_tp = 1
    config.mlp_bias = False
    config.attention_bias = False
    if not ofted:
        model = PcaLlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            train_key=False,
            train_value=False,
            config=config,
            *model_args,
            **kwargs
        ).to(device)

        return model

    def orthogonal_(X, dtype):
        # print(X.device)
        X = X.to(torch.float32)
        n = X.shape[-1]
        Id = torch.eye(n, dtype=X.dtype, device=X.device)
        
        A = X - X.mH
        Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
        
        return Q.to(dtype)

    head_dim = config.hidden_size // config.num_attention_heads
    unitary_transform = torch.load(os.path.join(checkpoint_dir, 'unitary_transform_weight.bin'))
    all_layers_mean_key_states = [torch.tensor(0) for _ in range(config.num_hidden_layers)]
    all_layers_mean_value_states = [torch.tensor(0) for _ in range(config.num_hidden_layers)]
    all_layers_key_states_eigenvectors_descending = [torch.tensor(0) for _ in range(config.num_hidden_layers)]
    all_layers_value_states_eigenvectors_descending = [torch.tensor(0) for _ in range(config.num_hidden_layers)]
    for name, param in unitary_transform.items():
        layer_idx = int(name.split('.')[-3])
        if str(name.split('.')[-1]) == "mean_key_weights":
            split_mean_key = torch.split(param, 1, dim=1)             # param.shape = [1, 32, 1, 128]  
            all_layers_mean_key_states[layer_idx] = [mean_key.view(head_dim).cpu().detach().clone() for mean_key in split_mean_key]
            
        if str(name.split('.')[-1]) == "mean_value_weights":
            split_mean_value = torch.split(param, 1, dim=1)             # param.shape = [1, 32, 1, 128]  
            all_layers_mean_value_states[layer_idx] = [mean_value.view(head_dim).cpu().detach().clone() for mean_value in split_mean_value]
        
        if str(name.split('.')[-1]) == "key_unitary_transform_weights":
            # split_key_unitary_transform = torch.split(param, 1, dim=0)
            ortho_param = orthogonal_(param, param.dtype)
            split_key_unitary_transform = torch.split(ortho_param, 1, dim=0)   # param.shape = [32, 128, 128]  
            all_layers_key_states_eigenvectors_descending[layer_idx] = [key_unitary_transform.squeeze(dim = 0).cpu().detach().clone() for key_unitary_transform in split_key_unitary_transform]
            
        if str(name.split('.')[-1]) == "value_unitary_transform_weights":
            # split_value_unitary_transform = torch.split(param, 1, dim=0) 
            ortho_param = orthogonal_(param, param.dtype)
            split_value_unitary_transform = torch.split(ortho_param, 1, dim=0)   # param.shape = [32, 128, 128]
            all_layers_value_states_eigenvectors_descending[layer_idx] = [value_unitary_transform.squeeze(dim = 0).cpu().detach().clone() for value_unitary_transform in split_value_unitary_transform]
            
    model = PcaLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        all_layers_mean_key_states = all_layers_mean_key_states, 
        all_layers_mean_value_states = all_layers_mean_value_states,
        all_layers_key_states_eigenvectors_descending = all_layers_key_states_eigenvectors_descending,
        all_layers_value_states_eigenvectors_descending = all_layers_value_states_eigenvectors_descending,
        train_key=train_key,
        train_value=train_value,
        key_truncate_index=key_truncate_index,
        value_truncate_index=value_truncate_index,
        config=config,
        *model_args,
        **kwargs
    ).to(device)

    
    if lora_trained:
        print("load PeftModel")
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')

    # 返回自定义模型实例
    return model


def repeat_unitary_transform(unitary_transform: torch.Tensor, n_rep: int):
    num_key_value_heads, head_dim1, head_dim2 = unitary_transform.shape
    if n_rep == 1:
        return unitary_transform
    unitary_transform = unitary_transform[:, None, :, :].expand(num_key_value_heads, n_rep, head_dim1, head_dim2).clone()
    return unitary_transform.reshape(num_key_value_heads * n_rep, head_dim1, head_dim2)

def repeat_mean_states(mean_states: torch.Tensor, n_rep: int):
    _, num_key_value_heads, _, head_dim = mean_states.shape
    if n_rep == 1:
        return mean_states
    mean_states = mean_states[:, :, None, :].expand(1, num_key_value_heads, n_rep, 1, head_dim).clone()
    return mean_states.reshape(1, num_key_value_heads * n_rep, 1, head_dim)

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
        
        # ones = torch.ones(1, self.num_key_value_heads, 1, self.head_dim)
        mean_key_states_tensor = torch.stack(mean_key_states, dim=0).unsqueeze(dim=0).unsqueeze(dim=2).to(self.dtype)
        mean_value_states_tensor = torch.stack(mean_value_states, dim=0).unsqueeze(dim=0).unsqueeze(dim=2).to(self.dtype)
        key_unitary_transform_matrix_tensor = self.make_unitary_transform_det_1(unitary_transform_matrix_tensor = torch.stack(key_unitary_transform_matrix, dim=0)).to(self.dtype)
        value_unitary_transform_matrix_tensor = self.make_unitary_transform_det_1(unitary_transform_matrix_tensor = torch.stack(value_unitary_transform_matrix, dim=0)).to(self.dtype)
        # print(mean_key_states_tensor.shape)
        # print(key_unitary_transform_matrix_tensor.shape)
        # print(self.num_heads)
        if self.num_key_value_groups != 1:
            if mean_key_states_tensor.shape[1] != self.num_heads:
                mean_key_states_tensor = repeat_mean_states(mean_key_states_tensor, self.num_key_value_groups)
                mean_value_states_tensor = repeat_mean_states(mean_value_states_tensor, self.num_key_value_groups)
            if key_unitary_transform_matrix_tensor.shape[0] != self.num_heads:
                key_unitary_transform_matrix_tensor = repeat_unitary_transform(key_unitary_transform_matrix_tensor, self.num_key_value_groups)
                value_unitary_transform_matrix_tensor = repeat_unitary_transform(value_unitary_transform_matrix_tensor, self.num_key_value_groups)



        self.mean_key_weights = nn.Parameter(mean_key_states_tensor) if train_key else None
        self.mean_value_weights = nn.Parameter(mean_value_states_tensor) if train_value else None 
        self.key_unitary_transform_weights = nn.Parameter(self.inverse_orthogonal_(key_unitary_transform_matrix_tensor, orthogonal_type='cayley')) if train_key else None       
        self.value_unitary_transform_weights = nn.Parameter(self.inverse_orthogonal_(value_unitary_transform_matrix_tensor, orthogonal_type='cayley')) if train_value else None

      
        self.merged_weights = False
        
        self.register_buffer('mean_key_states', mean_key_states_tensor)
        self.register_buffer('mean_value_states', mean_value_states_tensor)
        self.register_buffer('key_unitary_transform_matrix', key_unitary_transform_matrix_tensor)
        self.register_buffer('value_unitary_transform_matrix', value_unitary_transform_matrix_tensor)
        
        # self.mean_key_weights = nn.Parameter(self.mean_key_states) if train_key else None
        # self.mean_value_weights = nn.Parameter(self.mean_value_states) if train_value else None
        # self.key_unitary_transform_weights = nn.Parameter(self.inverse_orthogonal_(self.key_unitary_transform_matrix, orthogonal_type='cayley')) if train_key else None       
        # self.value_unitary_transform_weights = nn.Parameter(self.inverse_orthogonal_(self.value_unitary_transform_matrix, orthogonal_type='cayley')) if train_value else None
    def make_unitary_transform_det_1(self, unitary_transform_matrix_tensor):
        for head_idx in range(unitary_transform_matrix_tensor.shape[0]):
            if torch.det(unitary_transform_matrix_tensor[head_idx].to(torch.float32)) < 0:
                unitary_transform_matrix_tensor[head_idx][:, 0] = - unitary_transform_matrix_tensor[head_idx][:, 0]
        return unitary_transform_matrix_tensor
    
    
    # use to initialize unitary_transform_weights, which guarantees orthogonal_(initialized unitary_transform_weights) = unitary_transform_matrix we calculate through eigen-decomposition
    def inverse_orthogonal_(self, Q, orthogonal_type='cayley'):     
        Q = Q.to(torch.float32)
        if orthogonal_type=='cayley':
            n = Q.shape[-1]
            Id = torch.eye(n, dtype=Q.dtype, device=Q.device)
            A = 2 * (Q - Id) @ torch.linalg.inv(Id + Q)
            X = 0.5 * (A.tril(-1) - A.transpose(-1, -2).tril(-1))
            # X = torch.zeros_like(A)
            # X.triu_(1).copy_(-A.triu(1))
            # X.tril_(-1).copy_(A.tril(-1))
        elif orthogonal_type=='exp':
            Q_np = Q.cpu().numpy()
            if len(Q.shape) == 2:
                A_np = scipy.linalg.logm(Q_np)
                A = torch.from_numpy(A_np.real)
                X = 0.5 * (A.tril(-1) - A.transpose(-1, -2).tril(-1))
            elif len(Q.shape) == 3:
                A_np = np.empty_like(Q_np)
                batch_size, n, m = Q_np.shape
                if n != m:
                    raise ValueError("Each matrix in the batch must be square (128x128).")
                for i in range(batch_size):
                    A_np[i] = scipy.linalg.logm(Q_np[i])
    
                # Convert the result back to a torch tensor
                A = torch.from_numpy(A_np.real)
            return A
                
        elif orthogonal_type=='householder':
            pass
        else:
            raise ValueError(
                "orthogonal_type should be cayley or exp or householder"
            ) 
        return X.to(self.dtype)

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
    def merge_wights(self):         # bug, notice that self.value_unitary_transform_matrix is saved in initilization
        mean_value_states = self.mean_value_weights if self.train_value else self.mean_value_states
        value_unitary_transform_matrix = self.orthogonal_(self.value_unitary_transform_weights) if self.train_value else self.value_unitary_transform_matrix

        o_proj_matrix = self.o_proj.weight.data.t()
        weight = torch.einsum('hdc,hcb->hdb', value_unitary_transform_matrix.transpose(1, 2), o_proj_matrix.reshape(self.num_heads, self.head_dim, self.hidden_size))
        bias = torch.matmul(mean_value_states.view(self.config.hidden_size), o_proj_matrix)
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
        key_truncate_index: Union[int, torch.LongTensor] = 128,
        value_truncate_index: Union[int, torch.LongTensor] = 128, 
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
        key_unitary_transform_matrix = self.orthogonal_(self.key_unitary_transform_weights) if self.train_key else self.key_unitary_transform_matrix
        value_unitary_transform_matrix = self.orthogonal_(self.value_unitary_transform_weights) if self.train_value else self.value_unitary_transform_matrix
        # check_tensor(key_unitary_transform_matrix, 'key')
        # check_tensor(value_unitary_transform_matrix, 'value')
        # if self.training and self.merged_weights and self.train_value:             # when training, we need to update the weights of output_proj from time to time
        #     self.update_o_proj_weights(mean_value_states, value_unitary_transform_matrix)
            
        if not self.merged_weights and not self.training:            
            self.merge_wights()
            
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
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = unitary_transform(attention_states=query_states, unitary_transform_matrix=key_unitary_transform_matrix, truncate_index=torch.max(key_truncate_index).item() if isinstance(key_truncate_index, torch.LongTensor) else key_truncate_index)
        key_states = unitary_transform(attention_states=key_states, unitary_transform_matrix=key_unitary_transform_matrix, mean_states=mean_key_states, truncate_index=torch.max(key_truncate_index).item() if isinstance(key_truncate_index, torch.LongTensor) else key_truncate_index)
        value_states = unitary_transform(attention_states=value_states, unitary_transform_matrix=value_unitary_transform_matrix, mean_states=mean_value_states, truncate_index=torch.max(value_truncate_index).item() if isinstance(value_truncate_index, torch.LongTensor) else value_truncate_index)
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(                               # key: (bsz, num_heads, kv_len, truncated_dim < 128)   value: (bsz, num_heads, kv_len, truncated_dim = 128)
                key_states, 
                value_states, 
                self.layer_idx, 
                cache_kwargs, 
                truncate_key = True, 
                truncate_value = True, 
                key_truncate_index = key_truncate_index, 
                value_truncate_index = value_truncate_index,
            )
            # print(isinstance(past_key_value, PcaDynamicCache))


        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) + torch.matmul(original_query_states, mean_key_states.transpose(2, 3))) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        if isinstance(value_truncate_index, torch.LongTensor):
            attn_output = torch.einsum('bhld,hdc->bhlc', attn_output, value_unitary_transform_matrix[..., :torch.max(value_truncate_index).item()].transpose(1, 2)) + mean_value_states
        else:
            attn_output = torch.einsum('bhld,hdc->bhlc', attn_output, value_unitary_transform_matrix[..., :value_truncate_index].transpose(1, 2)) + mean_value_states
        attn_output = attn_output.transpose(1, 2).contiguous()               # (bsz, q_len, self.num_heads, self.head_dim < 128)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # # custom output_proj
        # if self.training:
        #     if isinstance(value_truncate_index, torch.LongTensor):
        #         attn_output = torch.einsum('bhld,hdc->bhlc', attn_output, value_unitary_transform_matrix[..., :torch.max(value_truncate_index).item()].transpose(1, 2)) + mean_value_states
        #     else:
        #         attn_output = torch.einsum('bhld,hdc->bhlc', attn_output, value_unitary_transform_matrix[..., :value_truncate_index].transpose(1, 2)) + mean_value_states
        #     attn_output = attn_output.transpose(1, 2).contiguous()               # (bsz, q_len, self.num_heads, self.head_dim < 128)
        #     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        #     attn_output = self.o_proj(attn_output)
        # else:
        #     attn_output = attn_output.transpose(1, 2).contiguous()               # (bsz, q_len, self.num_heads, self.head_dim < 128)
        #     attn_output = attn_output.reshape(bsz, q_len, -1)
        #     if isinstance(value_truncate_index, torch.LongTensor):
        #         attn_output = torch.matmul(attn_output, self.transformed_weight[:, :torch.max(value_truncate_index).item(), :].reshape(-1, self.hidden_size)) + self.transformed_bias 
        #     else:
        #         attn_output = torch.matmul(attn_output, self.transformed_weight[:, :value_truncate_index, :].reshape(-1, self.hidden_size)) + self.transformed_bias

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
        key_truncate_index: Union[int, torch.LongTensor] = 128,
        value_truncate_index: Union[int, torch.LongTensor] = 128, 
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
        key_truncate_index: Union[int, List[int], torch.LongTensor] = 128,
        value_truncate_index: Union[int, List[int], torch.LongTensor] = 128, 
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
        
        # if past_key_values is not None:
        #     print(isinstance(past_key_values, PcaDynamicCache))

        # if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs), here past_key_values is None, so after this building an empty 
        if use_cache:
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

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # print("before layer_forward", isinstance(past_key_values, PcaDynamicCache))
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    key_truncate_index[layer_idx] if isinstance(key_truncate_index, (list, torch.LongTensor)) else key_truncate_index,
                    value_truncate_index[layer_idx] if isinstance(key_truncate_index, (list, torch.LongTensor)) else value_truncate_index, 
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
                    key_truncate_index=key_truncate_index[layer_idx] if isinstance(key_truncate_index, (list, torch.LongTensor)) else key_truncate_index,
                    value_truncate_index=value_truncate_index[layer_idx] if isinstance(key_truncate_index, (list, torch.LongTensor)) else value_truncate_index, 
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            # print("after layer_forward", isinstance(past_key_values, PcaDynamicCache))
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            # print("next_decoder_cache", isinstance(next_decoder_cache, PcaDynamicCache))
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        # print("next_cache", isinstance(next_cache, PcaDynamicCache))
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
        self.train_key = config.train_key
        self.train_value = config.train_value
        self.key_truncate_index = config.key_truncate_index
        self.value_truncate_index = config.value_truncate_index


        self.model = PcaLlamaModel(
            config, 
            config.all_layers_mean_key_states,
            config.all_layers_mean_value_states,
            config.all_layers_key_unitary_transform_matrix, 
            config.all_layers_value_unitary_transform_matrix, 
            config.train_key,
            config.train_value,
        )
        config.all_layers_mean_key_states = None
        config.all_layers_mean_value_states = None
        config.all_layers_key_unitary_transform_matrix = None
        config.all_layers_value_unitary_transform_matrix = None
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        key_truncate_index = kwargs.get("key_truncate_index", cal_key_truncate_index)
        value_truncate_index = kwargs.get("value_truncate_index", cal_value_truncate_index)

        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                if isinstance(key_truncate_index, torch.LongTensor) or isinstance(value_truncate_index, torch.LongTensor):
                    cache_length = past_length = past_key_values[0][0][0].shape[1]
                else:
                    cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "key_truncate_index": key_truncate_index,
                "value_truncate_index": value_truncate_index,
            }
        )
        return model_inputs
    
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        all_layers_mean_key_states = None, 
        all_layers_mean_value_states = None,
        all_layers_key_states_eigenvectors_descending = None,
        all_layers_value_states_eigenvectors_descending = None,
        train_key = True,
        train_value = True,  
        key_truncate_index = None,
        value_truncate_index = None,
        *model_args, 
        **kwargs):

        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        config.all_layers_mean_key_states = all_layers_mean_key_states 
        config.all_layers_mean_value_states = all_layers_mean_value_states 
        config.all_layers_key_unitary_transform_matrix = all_layers_key_states_eigenvectors_descending 
        config.all_layers_value_unitary_transform_matrix = all_layers_value_states_eigenvectors_descending 

        config.train_key = train_key
        config.train_value = train_value

        config.key_truncate_index = key_truncate_index
        config.value_truncate_index = value_truncate_index

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
        key_truncate_index: Union[int, List[int], torch.LongTensor] = cal_key_truncate_index,                    # change this
        value_truncate_index: Union[int, List[int], torch.LongTensor] = cal_value_truncate_index,
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

        if self.key_truncate_index:
            try:
                key_truncate_index = torch.full((32, 32), int(self.key_truncate_index), dtype = torch.long).cpu()
            except:
                key_truncate_index = load_from_txt(self.key_truncate_index).cpu()
        else:
            raise ValueError("You must pass in a key truncate index")

        if self.value_truncate_index:
            try:
                value_truncate_index = torch.full((32, 32), int(self.value_truncate_index), dtype = torch.long).cpu()
            except:
                value_truncate_index = load_from_txt(self.value_truncate_index).cpu()
        else:
            raise ValueError("You must pass in a value truncate index")
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        logits = logits.float()             # (bsz, seq_len, 32000)

        loss = None
        if labels is not None:
            labels = labels.repeat(logits.shape[0] // labels.shape[0], 1)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            
            # print(shift_logits.shape, shift_labels.shape)       # torch.Size([3444, 32000]) torch.Size([574])
            loss = loss_fct(shift_logits, shift_labels)

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
