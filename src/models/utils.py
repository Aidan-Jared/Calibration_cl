
import re

import jax
import jax.numpy as jnp
import equinox as eqx


def tensor_to_jax(tensor):
    return jnp.array(tensor.detach().cpu().numpy())

def block_name_translator(path:tuple|str):
    name = path[1:]   
    name = re.sub(r'blocks\[(\d+)\]', r'blocks.\1', name)
    attn = re.compile(r"q_proj|k_proj|v_proj")
    if attn.search(name):
        name = re.sub(r"q_proj|k_proj|v_proj", 'qkv', name)
    if "fc" in name:
        name = re.sub(r'(blocks\.\d+)\.(fc)', r'\1.mlp.\2', name)
    return name

# only works on HuggingFace models from timm, can be adapted for other models but needs to be modified
def torch_to_equinox(model, state_dict, embedding_dim):
    leaves, treedef = jax.tree_util.tree_flatten_with_path(
        eqx.filter(model, eqx.is_inexact_array)
    )
    new_leaves = []
    name_change = re.compile(r"q_proj|k_proj|v_proj")
    for path, leaf in leaves:
        path = jax.tree_util.keystr(path)
        name = block_name_translator(path)
        if name in state_dict:
            if name_change.search(path):
                tensor = state_dict[name]
                if "q" in path:
                    tensor = tensor[:embedding_dim]
                    new_leaves.append(tensor_to_jax(tensor))
                elif "k" in path:
                    tensor = tensor[embedding_dim:embedding_dim*2]
                    new_leaves.append(tensor_to_jax(tensor))
                elif "v" in path:
                    tensor = tensor[embedding_dim*2:]
                    new_leaves.append(tensor_to_jax(tensor))
            else:
                new_leaves.append(tensor_to_jax(state_dict[name]))
        else:
            new_leaves.append(leaf)
    new_params = jax.tree_util.tree_unflatten(treedef, new_leaves)
    return eqx.combine(new_params, eqx.filter(model, lambda x: not eqx.is_inexact_array(x)))