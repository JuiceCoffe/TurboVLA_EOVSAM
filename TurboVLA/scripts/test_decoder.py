import json
import torch
from torch import nn
from turbovla.models.action_head import ObjectTransformerDecoderLayer, ObjectTransformerDecoder

torch.manual_seed(7)
base_layer = nn.TransformerDecoderLayer(32, 4, dim_feedforward=64, dropout=0.1, batch_first=True, norm_first=True)
base = nn.TransformerDecoder(base_layer, 3).eval()
custom = ObjectTransformerDecoder(ObjectTransformerDecoderLayer(32, 4, dim_feedforward=64, dropout=0.1, batch_first=True, norm_first=True), 3).eval()
custom.load_state_dict(base.state_dict(), strict=False)
x = torch.randn(2, 5, 32, requires_grad=True)
memory = torch.randn(2, 9, 32, requires_grad=True)
objects = torch.randn(2, 8, 32, requires_grad=True)
mask = nn.Transformer.generate_square_subsequent_mask(5)
padding = torch.tensor([[False] * 8 + [True], [False] * 9])
ref = base(x, memory, tgt_mask=mask, memory_key_padding_mask=padding)
plain = custom(x, memory, tgt_mask=mask, memory_key_padding_mask=padding)
assert torch.equal(ref, plain)
for layer in custom.layers:
    nn.init.zeros_(layer.object_attn.out_proj.weight)
    nn.init.zeros_(layer.object_attn.out_proj.bias)
zero_branch = custom(x, memory, tgt_mask=mask, memory_key_padding_mask=padding, object_memory=objects)
assert torch.equal(ref, zero_branch)
grad_ref = torch.autograd.grad(ref.sum(), (x, memory), retain_graph=True)
grad_zero = torch.autograd.grad(zero_branch.sum(), (x, memory), retain_graph=True)
assert all(torch.equal(a, b) for a, b in zip(grad_ref, grad_zero))
for layer in custom.layers:
    nn.init.xavier_uniform_(layer.object_attn.out_proj.weight)
out = custom(x, memory, object_memory=objects)
out.square().mean().backward()
assert objects.grad is not None and objects.grad.abs().sum() > 0
assert all(layer.object_attn.in_proj_weight.grad.abs().sum() > 0 for layer in custom.layers)
print(json.dumps({'baseline_output_max_error': (ref - plain).abs().max().item(),
                  'zero_extra_attention_max_error': (ref - zero_branch).abs().max().item(),
                  'baseline_gradients_exact': True, 'object_attention_gradients_all_layers': True}, indent=2))
