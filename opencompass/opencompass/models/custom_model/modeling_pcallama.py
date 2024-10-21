import math
import torch
import torch.nn.functional as F
import warnings
from .PcaCache import PcaDynamicCache
from typing import List, Optional, Tuple, Union
from transformers.models.llama.modeling_llama import *
from transformers.cache_utils import Cache, DynamicCache

def unitary_transform(attention_states: torch.Tensor, unitary_transform_matrix: torch.Tensor, mean_states: torch.Tensor = torch.tensor(0)):
    transformed_attention_states = torch.einsum('bhld,hdc->bhlc', (attention_states - mean_states), unitary_transform_matrix)
    return transformed_attention_states
    
class PcaLlamaAttention(LlamaAttention):
    def __init__(
        self, 
        config: LlamaConfig, 
        mean_key_states: List[torch.Tensor],
        mean_value_states: List[torch.Tensor],
        key_unitary_transform_matrix: List[torch.Tensor], 
        value_unitary_transform_matrix: List[torch.Tensor], 
        key_truncate_threshold: int = 0.2,
        value_truncate_threshold: int = 0.2,
        layer_idx: Optional[int] = None
    ):
        super().__init__(config, layer_idx)
        self.config = config
        
        # len(mean_key_states) = head_num = 32 -> (1, head_num, 1, 128)
        self.mean_key_states = torch.stack(mean_key_states, dim=0).unsqueeze(dim=0).unsqueeze(dim=2).to(self.dtype).cuda()           
        self.mean_value_states = torch.stack(mean_value_states, dim=0).unsqueeze(dim=0).unsqueeze(dim=2).to(self.dtype).cuda()  
        
        # len(self.unitary_transform_matrix) = head_num = 32 -> (head_num, 128, 128)
        self.key_unitary_transform_matrix = torch.stack(key_unitary_transform_matrix, dim=0).to(self.dtype).cuda()           
        self.value_unitary_transform_matrix = torch.stack(value_unitary_transform_matrix, dim=0).to(self.dtype).cuda()   
        self.key_truncate_threshold = key_truncate_threshold
        self.value_truncate_threshold = value_truncate_threshold

        self.merged_weights = False
        
    def merge_wights(self):
        original_matrix = self.o_proj.weight.data.t().cuda() 
        transformed_weight = torch.einsum('hdc,hcb->hdb', self.value_unitary_transform_matrix.transpose(1, 2), original_matrix.reshape(self.num_heads, self.head_dim, self.hidden_size)).reshape(self.hidden_size, self.hidden_size)
        bias = torch.matmul(self.mean_value_states.view(self.config.hidden_size), original_matrix)
        
        self.o_proj.weight.data = transformed_weight.t()
        self.o_proj.bias = nn.Parameter(bias)
        self.o_proj.bias.requires_grad = True
        
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
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        if not self.merged_weights:
            self.merge_wights()
            
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
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
        query_states = unitary_transform(attention_states=query_states, unitary_transform_matrix=self.key_unitary_transform_matrix)
        key_states = unitary_transform(attention_states=key_states, unitary_transform_matrix=self.key_unitary_transform_matrix, mean_states=self.mean_key_states)
        value_states = unitary_transform(attention_states=value_states, unitary_transform_matrix=self.value_unitary_transform_matrix, mean_states=self.mean_value_states)
                
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(                               # (bsz, num_heads, kv_len, truncated_dim < 128)
                key_states, value_states, self.layer_idx, cache_kwargs, 
                truncate_key = True, truncate_value = True, 
                key_truncate_threshold = self.key_truncate_threshold, value_truncate_threshold = self.value_truncate_threshold,
            )
            
        # value_states2 = unitary_transform(attention_states = value_states, unitary_transform_matrix = self.value_unitary_transform_matrix.transpose(-1, -2))
        # print(value_states2[:, :, -1:, :] - value_states1)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = (torch.matmul(query_states[..., :key_states.shape[-1]], key_states.transpose(2, 3)) + torch.matmul(original_query_states, self.mean_key_states.transpose(2, 3))) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

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
        key_truncate_threshold: int = 0.2,
        value_truncate_threshold: int = 0.2,
        layer_idx: Optional[int] = None
    ):
        super().__init__(config, layer_idx)
        self.self_attn = PcaLlamaAttention(
            config=config, 
            mean_key_states=mean_key_states,
            mean_value_states=mean_value_states,
            key_unitary_transform_matrix=key_unitary_transform_matrix, 
            value_unitary_transform_matrix=value_unitary_transform_matrix,
            key_truncate_threshold=key_truncate_threshold, 
            value_truncate_threshold=value_truncate_threshold, 
            layer_idx=layer_idx,
        )

class PcaLlamaModel(LlamaModel):
    def __init__(
        self, 
        config: LlamaConfig, 
        all_layers_mean_key_states: List[List[torch.Tensor]],                   
        all_layers_mean_value_states: List[List[torch.Tensor]],                   
        all_layers_key_unitary_transform_matrix: List[List[torch.Tensor]], 
        all_layers_value_unitary_transform_matrix: List[List[torch.Tensor]], 
        key_truncate_threshold: int = 0.2, 
        value_truncate_threshold: int = 0.2, 
        value_truncate_idx: List[int] = [127] * 32,  
    ):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [PcaLlamaDecoderLayer(
                config, 
                mean_key_states=all_layers_mean_key_states[layer_idx],
                mean_value_states=all_layers_mean_value_states[layer_idx],
                key_unitary_transform_matrix=all_layers_key_unitary_transform_matrix[layer_idx], 
                value_unitary_transform_matrix=all_layers_value_unitary_transform_matrix[layer_idx], 
                key_truncate_threshold=key_truncate_threshold, 
                value_truncate_threshold=value_truncate_threshold,
                layer_idx=layer_idx,
            ) for layer_idx in range(config.num_hidden_layers)]
        )
        self.value_truncate_idx = value_truncate_idx
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
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
            past_key_values = PcaDynamicCache.from_legacy_cache(head_dim = self.config.hidden_size // self.config.num_attention_heads, value_truncate_idx=self.value_truncate_idx, past_key_values = past_key_values)

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
        self.model = PcaLlamaModel(
            config, 
            config.all_layers_mean_key_states,
            config.all_layers_mean_value_states,
            config.all_layers_key_unitary_transform_matrix, 
            config.all_layers_value_unitary_transform_matrix, 
            config.key_truncate_threshold,
            config.value_truncate_threshold,
            config.value_truncate_idx,
        )