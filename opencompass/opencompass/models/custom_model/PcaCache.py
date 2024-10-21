import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from transformers.cache_utils import Cache, DynamicCache

def min_max_scale(tensor, min_value=0.0, max_value=10.0):
    # 获取张量的最小值和最大值
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    
    # 进行最小-最大归一化
    scaled_tensor = min_value + (tensor - tensor_min) * (max_value - min_value) / (tensor_max - tensor_min)
    
    return scaled_tensor

class PcaDynamicCache(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated, with additional support for PCA-based caching.

    It stores the Key and Value states as a list of tensors, one for each layer, and uses PCA to reduce the dimensionality.
    The expected shape for each tensor is `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, head_dim: int, value_truncate_idx: List[int] = [128]*32) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.value_truncate_idx = value_truncate_idx
        
        
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, with PCA dimension reduction.
        """
        if layer_idx < len(self):
            key_cache_states = self.key_cache[layer_idx]
            value_cache_states = self.value_cache[layer_idx]
            
            # bsz, num_heads, kv_len, compressed_key_dim = key_cache_states.size()
            bsz, num_heads, kv_len, compressed_value_dim = value_cache_states.size()

            # if compressed_key_dim < self.head_dim:
            #     pad_size = self.head_dim - compressed_key_dim
            #     pad = torch.zeros(bsz, num_heads, kv_len, pad_size, dtype=key_cache_states.dtype, device=key_cache_states.device)
            #     key_cache_states = torch.cat([key_cache_states, pad], dim=-1)
            if compressed_value_dim < self.head_dim:
                pad_size = self.head_dim - compressed_value_dim
                pad = torch.zeros(bsz, num_heads, kv_len, pad_size, dtype=value_cache_states.dtype, device=value_cache_states.device)
                value_cache_states = torch.cat([value_cache_states, pad], dim=-1)
                
            return (key_cache_states, value_cache_states)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        key_truncate_threshold: float = 0.2, 
        value_truncate_threshold: float = 0.2, 
        truncate_key: bool = False,
        truncate_value: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            if truncate_key:
                mean_key_states = torch.mean(key_states ** 2, dim=(0, 1, 2))
                # print(mean_key_states)
                truncate_indices = torch.where(mean_key_states < key_truncate_threshold)[0]
                key_truncate_index = truncate_indices[0] if len(truncate_indices) > 0 else key_states.shape[-1]
                key_truncate_index = 128
                # print(key_truncate_index)
                self.key_cache.append(key_states[..., :key_truncate_index])
            else:
                self.key_cache.append(key_states)
            
            if truncate_value:
                # mean_value_states = torch.mean(value_states ** 2, dim=(0, 1, 2))
                # print(mean_value_states)
                # scaled_states = min_max_scale(mean_value_states)
                # print(scaled_states)
                # truncate_indices = torch.where(mean_value_states > value_truncate_threshold)[0]
                # truncate_indices = torch.where(mean_value_states > torch.max(mean_value_states) * 0.05)[0]
                # value_truncate_index = truncate_indices[-1] if len(truncate_indices) > 0 else value_states.shape[-1]
                
                # value_truncate_index = 100

                value_truncate_index = self.value_truncate_idx[layer_idx]
                self.value_cache.append(value_states[..., :value_truncate_index])
            else:
                self.value_cache.append(value_states)
        else:
            if truncate_key:
                key_truncate_index = self.key_cache[layer_idx].shape[-1]
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states[..., :key_truncate_index]], dim=-2)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                
            if truncate_value:
                value_truncate_index = self.value_cache[layer_idx].shape[-1]
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states[..., :value_truncate_index]], dim=-2)
            else:
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.__getitem__(layer_idx = layer_idx)

    @classmethod
    def from_legacy_cache(cls, head_dim, value_truncate_idx: Optional[List[int]] = [127] * 32, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "PcaDynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls(head_dim, value_truncate_idx)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
