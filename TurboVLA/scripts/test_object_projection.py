import json
import torch
from turbovla.models.action_head import TurboVLAActionHead
from turbovla.models.configuration import ActionHeadConfig

torch.manual_seed(7)
head = TurboVLAActionHead(ActionHeadConfig(), 256, 8, 2048, object_feature_dim=1792, num_views=2)
projections = head.decoder.object_projections
assert len(projections) == 2
assert projections[0].weight.data_ptr() != projections[1].weight.data_ptr()
features = torch.randn(2, 400, 1792)
memory = torch.randn(2, 16, 256)
state = torch.randn(2, 8)
captured = []
hook = head.decoder.decoder.register_forward_pre_hook(lambda m, args, kwargs: captured.append(kwargs['object_memory'].detach().clone()), with_kwargs=True)
head.eval()
with torch.no_grad():
    head(memory, state, features)
    projections[0].weight.add_(0.1)
    head(memory, state, features)
assert not torch.equal(captured[0][:, :200], captured[1][:, :200])
assert torch.equal(captured[0][:, 200:], captured[1][:, 200:])
hook.remove()
head.train()
action = head(memory, state, features)
assert action.shape == (2, 12, 7) and torch.isfinite(action).all()
action.square().mean().backward()
gradients = [p.weight.grad.norm().item() for p in projections]
assert all(torch.isfinite(p.weight.grad).all() and g > 0 for p, g in zip(projections, gradients))
print(json.dumps({'independent_parameter_storage': True, 'view0_weight_change_leaves_view1_memory_identical': True,
                  'projection_shapes': [list(p.weight.shape) for p in projections],
                  'gradient_norms': gradients, 'action_shape': list(action.shape)}, indent=2))
