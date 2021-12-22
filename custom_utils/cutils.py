import torch
import numpy as np

def single_head_flops(length, config, ratio):
    L = length
    d = config.hidden_size
    # q, k, v: (L x d) (d x 64) -> (L x 64)
    flops_qkv = (L * 64 * d * 2) * 3
    # attn: (L x 64) (64 x L) -> (L x L)
    flops_attn = L * L * 64 * 2
    # attn * v: (L x L) (L x 64) -> (L x 64)
    flops_attn_v = L * 64 * L * 2
    # self_output: (L x 64) (64 x d) -> ( L x d)
    flops_self = L * d * 64 * 2
    return ratio * (flops_qkv + flops_attn + flops_attn_v + flops_self)

def ffn_flops(length, config, ratio):
    L = length
    d = config.hidden_size
    inter = config.intermediate_size * ratio # using ratio to get average number of intermediate nodes
    # ffn0: (L x d) (d x inter) -> (L x inter)
    flops_fc0 = L * inter * d * 2
    # ffn1: (L x inter) (inter x d) -> (L x d)
    flops_fc1 = L * d * inter * 2
    return flops_fc0 + flops_fc1

def head_mask_generator_flops(config):
    d = config.hidden_size
    # (1 x d) (d x 64) -> (1 x 64)
    flops_fc0 = 1 * 64 * d * 2
    # (1 x 64) (64 x 12) -> (1 x 12)
    flops_fc1 = 1 * 12 * 64 * 2
    return flops_fc0 + flops_fc1

def ffn_mask_generator_flops(config):
    d = config.hidden_size
    inter = config.intermediate_size
    # (1x d) (d x 64) -> (1 x 64)
    flops_fc0 = 1 * 64 * d * 2
    # (1 x 64) (64 x 4d) -> (1 x 3072)
    flops_fc1 = 1 * inter * 64 * 2
    return flops_fc0 + flops_fc1

def compute_model_flops(length, config, model=None, num_samples=1, is_sparsity=False):
    total_flops = 0.0

    if not is_sparsity:
        total_flops = single_head_flops(length, config, 1.0) * config.num_attention_heads
        total_flops += ffn_flops(length, config, 1.0) 
        total_flops *= config.num_hidden_layers
    elif model is not None:
        for key, value in model.named_buffers():
            if 'head_nopruned_times' in key:
                value = value.detach().cpu().numpy() / float(num_samples)
                for h in value:
                    total_flops += single_head_flops(length, config, h)
                    total_flops += head_mask_generator_flops(config)

            if 'ffn_nopruned_times' in key:
                value = np.mean(value.detach().cpu().numpy() / float(num_samples))

                total_flops += ffn_flops(length, config, value)
                total_flops += ffn_mask_generator_flops(config)

    return total_flops

def get_mean_times(model=None, num_samples=1):
    for key, value in model.named_buffers():
        if 'head_nopruned_times' in key:
            value = value.detach().cpu().numpy() / float(num_samples)
            print(np.sum(value))

        if 'ffn_nopruned_times' in key:
            value = value.detach().cpu().numpy() / float(num_samples)
            print(np.sum(value))

def compute_model_sparsity(length, num_samples, model, logger, writer, log):

    logger.info("***** sparsity and flops *****")
    config = model.config
    total_flops = compute_model_flops(length, config)
    total_compressed_flops = compute_model_flops(length, config, model, num_samples, True)

    original_flops = total_flops / (1024 ** 3)
    dynamic_flops = total_compressed_flops / (1024 ** 3)
    compress_ratio = total_compressed_flops / total_flops
    
    if log is not None:
        log({'compress_ratio': compress_ratio})

    logger.info("Original model flops = {:.3f} GFLOPs".format(original_flops))
    logger.info("Dynamic model average flops = {:.3f} GFLOPs".format(dynamic_flops))
    logger.info("Compressed ratio = {:.3f}".format(compress_ratio))

    if writer is not None:
        writer.write("Original model flops = {:.3f} GFLOPs\n".format(original_flops))
        writer.write("Dynamic model average flops = {:.3f} GFLOPs\n".format(dynamic_flops))
        writer.write("Compressed ratio = {:.3f}".format(compress_ratio))
    return compress_ratio

class Binarize_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, t):
        return (input > t).float()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None