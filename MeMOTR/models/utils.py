# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : To build a model.
import torch
import copy
import math
import torch.distributed
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import is_distributed, distributed_rank, is_main_process


def save_checkpoint(model: nn.Module, path: str, states: dict = None,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None):
    model = get_model(model)
    if is_main_process():
        save_state = {
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            'states': states
        }
        torch.save(save_state, path)
    else:
        pass
    return


def load_checkpoint(model: nn.Module, path: str, states: dict = None,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None):
    load_state = torch.load(path, map_location="cpu")

    if is_main_process():
        model.load_state_dict(load_state["model"])
    else:
        pass
    if optimizer is not None:
        optimizer.load_state_dict(load_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(load_state["scheduler"])
    if states is not None:
        states.update(load_state["states"])
    return


def get_activation_layer(activation: str):
    if activation == "ReLU":
        return nn.ReLU(True)
    elif activation == "GELU":
        return nn.GELU()
    else:
        raise ValueError(f"Do not support activation layer: {activation}")


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def get_model(model):
    return model if is_distributed() is False else model.module


# I think I do not use this function at all...
def query_masks_to_attn_mask(query_mask: torch.Tensor, n_heads: int, src_len: int):
    attn_mask = torch.ones((query_mask.shape[0], 1, query_mask.shape[1], query_mask.shape[1]),
                           dtype=torch.bool,
                           device=query_mask.device)
    for b in range(query_mask.shape[0]):
        usefull_length = sum(~query_mask[b]).item()
        attn_mask[b, :, :usefull_length, :usefull_length] = False
    attn_mask = attn_mask.repeat(1, n_heads, 1, 1)
    attn_mask = attn_mask.reshape(query_mask.shape[0]*n_heads, query_mask.shape[1], query_mask.shape[1])
    return attn_mask


def pos_to_pos_embed(pos, num_pos_feats: int = 64, temperature: int = 10000, scale: float = 2 * math.pi):
    pos = pos * scale
    dim_i = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_i = temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / num_pos_feats)
    pos_embed = pos[..., None] / dim_i      # (N, M, n_feats) or (B, N, M, n_feats)
    pos_embed = torch.stack((pos_embed[..., 0::2].sin(), pos_embed[..., 1::2].cos()), dim=-1)
    pos_embed = torch.flatten(pos_embed, start_dim=-3)
    
    return pos_embed

def pos_to_pos_embed_rotated(pos, num_pos_feats: int = 64, temperature: int = 10000, scale: float = 2 * math.pi, version='le135'):
    """
    Args:
        pos: Tensor[..., 5], 最后维度依次是 (x, y, w, h, theta)
        num_pos_feats: 每个空间维度的 sin‑cos 特征数
        temperature: 频率控制因子
        scale: 原始坐标乘上 scale 再做三角编码，通常 = 2π
        version: 角度标准化方式，可选 'le135' 或其它（默认为 2π 归一）

    Returns:
        Tensor[..., 4 * num_pos_feats + 2]
    """
    # --- 多频 sin‑cos 编码 for x,y,w,h ---
    # (1) 缩放空间坐标
    pos4 = pos[..., :4] * scale               # [...,4]
    # (2) 生成频率因子
    dim_i = torch.arange(num_pos_feats, device=pos.device, dtype=torch.float32)
    dim_i = temperature ** (2 * (dim_i // 2) / num_pos_feats)  # [num_pos_feats]
    # (3) 广播相除，得到 [...,4, num_pos_feats]
    embed4 = pos4[..., None] / dim_i
    # (4) 偶数维度 sin，奇数维度 cos，再 stack
    sin4 = embed4[..., 0::2].sin()            # [...,4, num_pos_feats//2]
    cos4 = embed4[..., 1::2].cos()            # [...,4, num_pos_feats//2]
    embed4 = torch.stack((sin4, cos4), dim=-1) # [...,4, num_pos_feats//2, 2]
    # (5) flatten 成 [..., 4 * num_pos_feats]
    embed4 = torch.flatten(embed4, start_dim=-3, end_dim=-1)

    # --- 单频 sin‑cos 编码 for theta ---
    theta = pos[..., 4]  # [...], 旋转角，假设单位为弧度
    if version == 'le135':
        theta_scaled = theta * math.pi - math.pi / 4
    else:
        raise NotImplementedError(f"Unsupported version: {version}")
    
    # 单频 做 sin/cos → [..., 2]
    sin_t = theta_scaled.sin()
    cos_t = theta_scaled.cos()
    embed_theta = torch.stack((sin_t, cos_t), dim=-1)

    # --- 拼接并返回 ---
    # [..., 4*num_pos_feats] + [...,2] → [..., 4*num_pos_feats+2]
    return torch.cat((embed4, embed_theta), dim=-1)

def load_pretrained_model(model: nn.Module, pretrained_path: str, show_details: bool = False):
    if not is_main_process():
        return model
    pretrained_checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    pretrained_state_dict = pretrained_checkpoint["model"]
    model_state_dict = model.state_dict()

    pretrained_keys = list(pretrained_state_dict.keys())
    for k in pretrained_keys:
        if k in model_state_dict:
            if model_state_dict[k].shape != pretrained_state_dict[k].shape:
                if "class_embed" in k:
                    if model_state_dict[k].shape[0] == 1:
                        pretrained_state_dict[k] = pretrained_state_dict[k][1:2]
                    elif model_state_dict[k].shape[0] == 2:
                        pretrained_state_dict[k] = pretrained_state_dict[k][1:3]
                    elif model_state_dict[k].shape[0] == 3:
                        pretrained_state_dict[k] = pretrained_state_dict[k][1:4]
                    elif model_state_dict[k].shape[0] == 8:     # BDD100K
                        pretrained_state_dict[k] = model_state_dict[k]
                        # We directly do not use the pretrained class embed for BDD100K
                    else:
                        raise NotImplementedError('invalid shape: {}'.format(model_state_dict[k].shape))
                    
                # for bbox_embed:
                elif "bbox_embed" in k:
                    if len(pretrained_state_dict[k]) != len(model_state_dict[k]):
                        # missmatch for model need 5 out_ch
                        if 'weight' in k:
                            model_state_dict[k][:4, :] = pretrained_state_dict[k]
                            pretrained_state_dict[k] = model_state_dict[k]
                        elif 'bias' in k:
                            model_state_dict[k][:4] = pretrained_state_dict[k]
                            pretrained_state_dict[k] = model_state_dict[k]
                        else:
                            raise NotImplementedError(f"some thing wrong with the bbox_embed.")
                elif "ref_point_head" in k or "query_pos_head" in k:
                    model_state_dict[k][:,:512] = pretrained_state_dict[k]
                    pretrained_state_dict[k] = model_state_dict[k]
                    print(f'load the first 512 dim of {k} from pretrained model')
                else:
                    print(f"Parameter {k} has shape{pretrained_state_dict[k].shape} in pretrained model, "
                          f"but get shape{model_state_dict[k].shape} in current model.")
        elif "query_embed" in k:
            if pretrained_state_dict[k].shape == model_state_dict["det_query_embed"].shape:
                pretrained_state_dict["det_query_embed"] = pretrained_state_dict[k].clone()
            else:
                print(f"Det Query shape is not equal. Check if you turn on 'USE_DAB'.")
                pretrained_state_dict["det_query_embed"] = model_state_dict["det_query_embed"]
            del pretrained_state_dict[k]
        elif "tgt_embed" in k:  # for DAB
            if pretrained_state_dict[k].shape == model_state_dict["det_query_embed"].shape:
                pretrained_state_dict["det_query_embed"] = pretrained_state_dict[k].clone()
            else:
                pretrained_state_dict["det_query_embed"] = model_state_dict["det_query_embed"]
            del pretrained_state_dict[k]
        elif "refpoint_embed" in k:
            pre_shape =  pretrained_state_dict[k].shape
            model_shape = model_state_dict["det_anchor"].shape
            if pre_shape[0] == model_shape[0]:
                model_state_dict["det_anchor"][:,:4] = pretrained_state_dict[k].clone()
                pretrained_state_dict[k] = model_state_dict["det_anchor"]
            else:
                pretrained_state_dict["det_anchor"] = model_state_dict["det_anchor"]
                print(f"Pretrain model's query num is {pretrained_state_dict[k].shape[0]}, "
                      f"current model's query num is {model_state_dict['det_anchor'].shape[0]}, "
                      f"do not load these parameters.")
            del pretrained_state_dict[k]
        elif "backbone" in k:
            new_k = k[15:]
            new_k = "backbone.backbone.backbone" + new_k
            pretrained_state_dict[new_k] = pretrained_state_dict[k].clone()
            del pretrained_state_dict[k]
        elif "input_proj" in k:
            new_k = k[10:]
            new_k = "feature_projs" + new_k
            pretrained_state_dict[new_k] = pretrained_state_dict[k].clone()
            del pretrained_state_dict[k]

        else:
            pass

    not_in_model = 0
    for k in pretrained_state_dict:
        if k not in model_state_dict:
            not_in_model += 1
            if show_details:
                print(f"Parameter {k} in the pretrained model but not in the current model.")

    not_in_pretrained = 0
    for k in model_state_dict:
        if k not in pretrained_state_dict:
            pretrained_state_dict[k] = model_state_dict[k]
            not_in_pretrained += 1
            if show_details:
                print(f"There is a new parameter {k} in the current model, but not in the pretrained model.")

    # 检查所有不匹配的参数 
    for k, v in pretrained_state_dict.items():
        if v not in model_state_dict:
            print(f"Skip loading parameter {k}, not in model, loaded shape {v.shape}.")
        elif v.shape != model_state_dict[k].shape:
            print(f"Skip loading parameter {k}, required shape {model_state_dict[k].shape}, loaded shape {v.shape}.")


    model.load_state_dict(state_dict=pretrained_state_dict, strict=False)
    print(f"Pretrained model is loaded, there are {not_in_model} parameters droped "
          f"and {not_in_pretrained} parameters unloaded, set 'show details' True to see more details.")

    return model


def logits_to_scores(logits: torch.Tensor):
    return logits.sigmoid()
