import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
from transformers.cache_utils import Cache, DynamicCache


class PcaDynamicCache(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated, with additional support for PCA-based caching.

    It stores the Key and Value states as a list of tensors, one for each layer, and uses PCA to reduce the dimensionality.
    The expected shape for each tensor is `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        
        
    def __getitem__(self, layer_idx: int, key_truncate_index: torch.LongTensor = None, value_truncate_index: torch.LongTensor = ModuleNotFoundError) -> Tuple[torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, with PCA dimension reduction.
        """
        if layer_idx < len(self) and key_truncate_index is not None and value_truncate_index is not None:
            past_key_states = self.padding_key_value_states(reduced_attention_states=self.key_cache[layer_idx], truncate_index=key_truncate_index)
            past_value_states = self.padding_key_value_states(reduced_attention_states=self.value_cache[layer_idx], truncate_index=value_truncate_index)
            return (past_key_states, past_value_states)
            # return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        elif layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def truncate_key_value_states(self, attention_states: torch.Tensor, truncate_index: torch.LongTensor) -> List[torch.tensor]:
        reduced_attention_states = []
        for head_idx, trunc_idx in enumerate(truncate_index):
            reduced_attention_states.append(attention_states[:, head_idx, :, :trunc_idx])
        return reduced_attention_states
    
    def padding_key_value_states(self, reduced_attention_states: List[torch.Tensor], truncate_index: torch.LongTensor) -> torch.Tensor:
        max_dim = torch.max(truncate_index).item()
        padded_states = [
            torch.nn.functional.pad(state, (0, max_dim - trunc_idx))
            for state, trunc_idx in zip(reduced_attention_states, truncate_index)
        ]

        attention_states = torch.stack(padded_states, dim=1)
        return attention_states


    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        key_truncate_index: Union[int, torch.LongTensor] = torch.full((32, ), 128, dtype=torch.long),           # input key_states is just the padding key_states, we need to truncate them, 32 represent the head_num
        value_truncate_index: Union[int, torch.LongTensor] = torch.full((32, ), 128, dtype=torch.long), 
        truncate_key: bool = False,
        truncate_value: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokensx
        if layer_idx == 0:
            if isinstance(key_states, list):
                self._seen_tokens += key_states[0].shape[-2]
            else:
                self._seen_tokens += key_states.shape[-2]
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            if truncate_key and isinstance(key_truncate_index, torch.LongTensor):
                self.key_cache.append(self.truncate_key_value_states(attention_states=key_states, truncate_index=key_truncate_index))
            else:
                self.key_cache.append(key_states)
                  
            if truncate_value and isinstance(value_truncate_index, torch.LongTensor):
                self.value_cache.append(self.truncate_key_value_states(attention_states=value_states, truncate_index=value_truncate_index))
            else:
                self.value_cache.append(value_states)
        else:
            if truncate_key and isinstance(key_truncate_index, torch.LongTensor):
                for head_idx in range(len(self.key_cache[0])):
                    self.key_cache[layer_idx][head_idx] = torch.cat([self.key_cache[layer_idx][head_idx], key_states[:, head_idx, :, :self.key_cache[layer_idx][head_idx].shape[-1]]], dim=1)
            else:
                if truncate_key:
                    key_truncate_index = self.key_cache[layer_idx].shape[-1]
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states[..., :key_truncate_index]], dim=-2)
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                
            if truncate_value and isinstance(value_truncate_index, torch.LongTensor):
                for head_idx in range(len(self.value_cache[0])):
                    self.value_cache[layer_idx][head_idx] = torch.cat([self.value_cache[layer_idx][head_idx], value_states[:, head_idx, :, :self.value_cache[layer_idx][head_idx].shape[-1]]], dim=1)
            else:
                if truncate_value:
                    value_truncate_index = self.value_cache[layer_idx].shape[-1]
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states[..., :value_truncate_index]], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if isinstance(key_truncate_index, torch.LongTensor) or isinstance(value_truncate_index, torch.LongTensor):
            return self.__getitem__(layer_idx = layer_idx, key_truncate_index = key_truncate_index, value_truncate_index = value_truncate_index)
        else:
            return self.__getitem__(layer_idx = layer_idx)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        
        if isinstance(self.key_cache, list):
            return self.key_cache[layer_idx][0].shape[-2]
        else:
            return self.key_cache[layer_idx].shape[-2]
    
    @classmethod
    def from_legacy_cache(cls, head_dim, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "PcaDynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls(head_dim)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
