import os

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from src.models.resnet18 import Drop_Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class BasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    shortcut: list | None

    bn1: eqx.nn.GroupNorm
    bn2: eqx.nn.GroupNorm
    dropout: Drop_Path

    def __init__(
        self,
        in_chanels: int,
        out_chanels: int,
        stride: int = 1,
        dropout: float = 0.1,
        dtype=jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        subkey1, subkey2, subkey3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(
            in_chanels,
            out_chanels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dtype=dtype,
            key=subkey1,
        )
        self.conv2 = eqx.nn.Conv2d(
            out_chanels, 
            out_chanels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            dtype=dtype,
            key=subkey2
        )

        self.bn1 = eqx.nn.GroupNorm(8,out_chanels)
        self.bn2 = eqx.nn.GroupNorm(8,out_chanels)
        self.dropout = Drop_Path(dropout)

        if stride != 1 or in_chanels != out_chanels:
            self.shortcut = [
                eqx.nn.Conv2d(
                    in_chanels,
                    out_chanels,
                    kernel_size=1,
                    stride=stride,
                    use_bias=False,
                    dtype=dtype,
                    key=subkey3,
                ),
                eqx.nn.GroupNorm(8, out_chanels),
            ]
        else:
            self.shortcut = None

    def __call__(
        self,
        x: Float[Array, " batch c w h"],
        state: PyTree,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, PyTree]:

        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = jax.nn.elu(out)

        out = self.conv2(out)
        out, state = self.bn2(out, state)
        out = jax.nn.elu(out)

        out = self.dropout(out, key=key)

        if self.shortcut is not None:
            identity = self.shortcut[0](x)
            identity, state = self.shortcut[1](identity, state)
        else:
            identity = x

        # def _shortcut(x, state):
        #     i = self.shortcut[0](x)
        #     i, s = self.shortcut[1](i, state) 
        #     return i, s

        # identity, state = jax.lax.cond(
        #     self.shortcut is not None, _shortcut, lambda x, s: (x, s), x, state

        # )

        out = out + identity

        out = jax.nn.elu(out)
        return out, state


class ResNet32(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.GroupNorm
    layer1: eqx.nn.Sequential
    layer2: eqx.nn.Sequential
    layer3: eqx.nn.Sequential
    
    # fc: eqx.nn.Linear

    hidden_channels: int = eqx.field(static= True)
    dropout: float = eqx.field(static=True)

    avgpool: eqx.nn.AdaptiveAvgPool2d

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 16,
        num_classes: int = 10,
        dropout: float = .1,
        dtype=jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.conv1 = eqx.nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            dtype=dtype,
            key=subkey1,
        )
        self.bn1 = eqx.nn.GroupNorm(8, hidden_channels)

        self.layer1 = self._make_layer(
            hidden_channels, num_blocks=5, stride=1, dtype=dtype, key=subkey2
        )
        self.layer2 = self._make_layer(
            hidden_channels * 2, num_blocks=5, stride=2, dtype=dtype, key=subkey3
        )
        self.layer3 = self._make_layer(
            hidden_channels * 4, num_blocks=5, stride=2, dtype=dtype, key=subkey4
        )

        self.avgpool = eqx.nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = eqx.nn.Linear(hidden_channels * 8, num_classes, key=subkey6)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int, dtype, key: PRNGKeyArray
    ):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for stride in strides:
            key, subkey = jax.random.split(key)
            layers.append(
                BasicBlock(self.hidden_channels, out_channels, stride, dropout = self.dropout, dtype=dtype, key=subkey)
            )
            self.hidden_channels = out_channels
        return eqx.nn.Sequential(layers)

    def __call__(
        self,
        x: Float[Array, "batch c w h"],
        state: PyTree,
        *,
        key: PRNGKeyArray,
    ):
        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = jax.nn.elu(out)

        for block in self.layer1:
            key, subkey = jax.random.split(key)
            out, state = block(x=out, state=state, key=subkey)
        for block in self.layer2:
            key, subkey = jax.random.split(key)
            out, state = block(x=out, state=state, key=subkey)
        for block in self.layer3:
            key, subkey = jax.random.split(key)
            out, state = block(x=out, state=state, key=subkey)

        out = self.avgpool(out)
        out = out.reshape(
            out.shape[0]
        )

        # out = self.fc(out)

        return out, state

class singleHeadResNet32(eqx.Module):
    resnet: ResNet32
    fc: eqx.nn.Linear
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 16,
        num_classes: int = 10,
        num_splits: int = 0, #not used but here for compatibility
        dropout: float = .1,
        dtype=jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        subkey1, subkey2 = jax.random.split(key)
        self.resnet = ResNet32(input_channels, hidden_channels, num_classes, dropout, dtype, key=subkey1)
        self.fc = eqx.nn.Linear(hidden_channels * 4, num_classes + 1, dtype=dtype, key=subkey2)
        
    def __call__(
        self,
        x: Float[Array, "batch c w h"],
        state: PyTree,
        # task: int, # not used but here for compatibility
        *,
        key: PRNGKeyArray,
    ):
        out, state = self.resnet(x, state, key=key)
        out = self.fc(out)
        return out, state

class multiHeadResNet32(eqx.Module):
    resnet: ResNet32
    heads: list[eqx.nn.Linear]
    datatype: jnp.dtype = eqx.field(static=True)
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 16,
        num_classes: int = 10,
        num_splits: int = 5,
        dropout: float = .1,
        dtype=jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        self.datatype = dtype
        subkey1, subkey2 = jax.random.split(key)
        self.resnet = ResNet32(input_channels, hidden_channels, num_classes, dropout, dtype, key=subkey1)
        self.heads = [eqx.nn.Linear(hidden_channels * 4, num_classes // num_splits, dtype=dtype, key = subkey2)]
    
    def __call__(
        self,
        x: Float[Array, "batch c w h"],
        state: PyTree,
        task: int | None,
        *,
        key: PRNGKeyArray,
    ):
        out, state = self.resnet(x, state, key=key)
        if task is None:
            out = jnp.concatenate([h(out) for h in self.heads], axis=-1)
        else:
            out = self.heads[task](out)            
            # out = [h(out) for h in self.heads][task]
        return out, state
    
    def add_head(self, num_classes: int, *, key: PRNGKeyArray) -> None:
        subkey = jax.random.fold_in(key, len(self.heads))
        new_head = eqx.nn.Linear(self.heads[0].in_features, num_classes, dtype=self.datatype, key=subkey)
        
        new_head = kaiming_init_model(new_head, subkey)
        
        new_heads = self.heads + [new_head]
        return eqx.tree_at(lambda m: m.heads, self, new_heads)
        

class expandingHeadResNet(eqx.Module):
    resnet: ResNet32
    heads: list[eqx.nn.Linear]
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 16,
        num_classes: int = 10,
        num_splits: int = 5, 
        dtype=jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        subkey1, subkey2 = jax.random.split(key)
        self.resnet = ResNet32(input_channels, hidden_channels, num_classes, dtype, key=subkey1)
        self.head = eqx.nn.Linear(hidden_channels * 4, num_classes // num_splits, dtype=dtype, key=subkey2)
    
    def __call__(
        self,
        x: Float[Array, "batch c w h"],
        state: PyTree,
        task: int,
        *,
        key: PRNGKeyArray,
    ):
        out, state = self.resnet(x, state, key=key)
        out = self.head(out)
        return out, state
    
    def expand_head(self, num_classes: int, *, key: PRNGKeyArray):
        old_w = self.head.weight
        old_b = self.head.bias
        new_head = eqx.nn.Linear(self.head.in_features, old_w.shape[0] + num_classes, dtype=jnp.float32, key=key)
        
        new_w = jnp.concatenate([
            old_w,
            jax.nn.initializers.he_normal()(key, (num_classes, old_w.shape[1]), jnp.float32)
        ], axis = 0)
        
        if old_b is not None:
            new_b = jnp.concatenate([
                old_b,
                jnp.zeros(num_classes, dtype=jnp.float32)
            ])
        else:
            new_b = None
        
        eqx.tree_at(lambda h: h.weight, new_head, new_w)
        if new_b is not None:
            eqx.tree_at(lambda h: h.bias, new_head, new_b)
        
        eqx.tree_at(lambda m: m.head, self, new_head)

def kaiming_init_model(model, key):
    leaves, treedef = jax.tree_util.tree_flatten(model)
    keys = jax.random.split(key, len(leaves))
    
    def reinit(leaf, k):
        if hasattr(leaf, "shape") and leaf.ndim >= 2:
            return jax.nn.initializers.he_normal()(k, leaf.shape, leaf.dtype)
        return leaf
    
    new_leaves = [reinit(leaf, k) for leaf, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)