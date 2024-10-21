import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
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
        
        
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, with PCA dimension reduction.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        key_truncate_index: int = 128, 
        value_truncate_index: int = 128, 
        truncate_key: bool = False,
        truncate_value: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            if truncate_key:
                self.key_cache.append(key_states[..., :key_truncate_index])
            else:
                self.key_cache.append(key_states)
            
            if truncate_value:
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
    def from_legacy_cache(cls, head_dim, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "PcaDynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls(head_dim)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
