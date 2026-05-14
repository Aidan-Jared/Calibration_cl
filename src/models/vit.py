import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Array, PRNGKeyArray, PyTree

from src.models.utils import torch_to_equinox

from src.models.resnet18 import Drop_Path
from typing import Callable

from timm import create_model

class Attention(eqx.Module):
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    proj: nn.Linear
    attn_dropout: nn.Dropout
    proj_dropout: nn.Dropout
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(
            self, 
            dim: int,
            num_heads: int,
            key: PRNGKeyArray,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.    
        ):

        subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -.5

        self.q_proj = nn.Linear(dim, dim, use_bias=qkv_bias, key=subkey1)
        self.k_proj = nn.Linear(dim, dim, use_bias=qkv_bias, key=subkey2)
        self.v_proj = nn.Linear(dim, dim, use_bias=qkv_bias, key=subkey3)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, key=subkey4)
        self.proj_dropout = nn.Dropout(proj_drop)

    def _shape(
            self,
            array: Array,
            seq_len: int,
            bsz: int
    ) -> Array:
        return array.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
    
    def __call__(
            self,
            x: Array,
            *,
            key: PRNGKeyArray| None = None
    ) -> Array:
        subkey1, subkey2 = jax.lax.cond(
            key is not None,
            lambda k: jax.random.split(k),
            lambda: (None, None),
            key
        )
        
        B, N, C = x.shape
        
        q = self._shape(self.q_proj(x), N, B).reshape(B * self.num_heads, -1, self.head_dim)
        k = self._shape(self.k_proj(x), -1, B).reshape(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).reshape(B * self.num_heads, -1, self.head_dim)

        attn_weights = jnp.einsum("bdq, bdk -> bqk", q, k) * self.scale

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = self.attn_dropout(attn_weights, key = subkey1)

        attn_output = jnp.einsum("bqk,bdv-> bdk", attn_probs, v)

        attn_output = attn_output.reshape(B, self.num_heads, N, self.head_dim).transpose(1,2).reshape(B,N,C)
        
        x = self.proj(attn_output)
        x = self.proj_dropout(x, key = subkey2)

        return x

class Block(eqx.Module):
    norm1: nn.LayerNorm
    attn: Attention
    norm2: nn.LayerNorm
    fc1: nn.Linear
    fc2: nn.Linear
    mlp_drop: nn.Dropout
    drop_path: Drop_Path

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float,
            key: PRNGKeyArray,
            qkv_bias: bool = False,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.
    ):
        subkey1, subkey2, subkey3 = jax.random.split(key,3)

        self.drop_path = Drop_Path(drop_path)

        self.norm1 = nn.LayerNorm(dim)

        self.attn = Attention(dim, num_heads,subkey1, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden, key=subkey2)
        self.fc2 = nn.Linear(mlp_hidden, dim, key=subkey3)
        self.mlp_drop = nn.Dropout(drop)

    def __call__(
            self,
            x: Array,
            state: nn._stateful.State,
            *,
            key: PRNGKeyArray | None = None,
    ) -> tuple[Array, nn._stateful.State]:
        
        subkey1, subkey2, subkey3, subkey4, subkey5, key = jax.lax.cond(
            key is not None,
            lambda k: jax.random.split(k, 6),
            lambda: (None, None, None, None, None, None),
            key
        )
        x, state = self.norm1(x, state)
        x = x + self.drop_path(self.attn(x, key = subkey1), key = subkey2)
        residual = x

        x, state = self.norm2(x, state)
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.mlp_drop(x, key = subkey3)

        x = self.fc2(x)
        x = self.mlp_drop(x, key = subkey4)
        x = self.drop_path(x, key = subkey5)

        
        x = x + residual
        return x, state

class PatchEmbed(eqx.Module):
    proj: nn.Conv2d
    norm: nn.LayerNorm | nn.Identity
    patch_size: int
    img_size: int
    num_patches: int
    embed_dim: int

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            embed_dim: int = 768,
            norm_layer: bool = False,
            *,
            key: PRNGKeyArray
        ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size)**2
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            key=key
        )

        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def __call__(
            self,
            x: Array
    ):
        x = self.proj(x)
        x = x.reshape(self.embed_dim, -1).T

        x = jax.vmap(self.norm)(x)
        return x

class VisionTransformer(eqx.Module):
    num_classes: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    patch_embed: PatchEmbed
    cls_token: jax.Array
    dist_token: jax.Array
    pos_embed: jax.Array
    pos_drop: nn.Dropout
    blocks: list[Block]
    norm: nn.LayerNorm
    head: nn.Linear | nn.Identity
    # head_dist: nn.Linear
    # embeddings: list[jax.Array]
    adapter_list: list
    adapter_gates: list

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chan: int = 3,
            num_classes: int = 1000,
            embed_dim: int = 1024,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            tunning_config: dict | None = None,
            *,
            key: PRNGKeyArray,
    ):
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chan, embed_dim, key=subkey1)

        self.cls_token = jnp.zeros((1,1, embed_dim), dtype=jnp.float32)
        self.dist_token = jnp.zeros((1,1, embed_dim), dtype=jnp.float32)
        self.pos_embed = jnp.zeros((1, self.patch_embed.num_patches + 2, embed_dim), dtype=jnp.float32)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in jnp.linspace(0, drop_path_rate, depth)]
        key, *subkeys = jax.random.split(subkey2, depth+1)

        self.blocks = [
            Block(
                embed_dim, num_heads, mlp_ratio, subkeys[i], qkv_bias, 
                drop_rate, attn_drop_rate, dpr[i]
            ) for i in range(depth)
        ]

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes, key=subkey3) if num_classes > 0 else nn.Identity()

        # self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # self.embeddings = [
        #     jnp.ones((1, tunning_config["vpt_num"], embed_dim), dtype=jnp.float32) for _ in range(depth)
        # ]
        
    def __call__(
            self,
            x: Array,
            state,
            *,
            key: PRNGKeyArray | None = None
    ) -> tuple[Array, nn._stateful.State]:
        
        key, subkey = jax.lax.cond(
            key is not None,
            lambda k: jax.random.split(k),
            lambda: (None, None),
            key
        )
        x = self.patch_embed(x)

        x = jnp.concat((self.cls_token, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x, key = subkey)

        for idx, blk in enumerate(self.blocks):
            key, subkey = jax.random.split(key)
            # x = jnp.concat(self.embeddings[idx], x, dim=1)

            x, state = blk(x, state, key = subkey)
        x, state = self.norm(x, state)
        outcome = x[:,0]

        return outcome, state
    
if __name__ == "__main__":
    seed = 42
    key = jax.random.PRNGKey(seed)
    subkey1, subkey2 = jax.random.split(key)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, key = subkey1)

    checkpoint_model = create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    
    new_model = torch_to_equinox(model, state_dict, 768)
    print("hi")