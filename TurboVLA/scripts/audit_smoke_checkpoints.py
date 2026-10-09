import hashlib
import json
import argparse
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
folder = Path(args.checkpoint_dir)
one = torch.load(folder / 'turbovla_step_1.pth', map_location='cpu', weights_only=False)
two = torch.load(folder / 'turbovla_step_2.pth', map_location='cpu', weights_only=False)
assert one['global_step'] == 1 and two['global_step'] == 2
changes = {}
for key, value in two['model_state_dict'].items():
    if 'object_projections' in key or 'object_attn.in_proj_weight' in key:
        delta = (value - one['model_state_dict'][key]).abs().max().item()
        assert delta > 0 and torch.isfinite(value).all()
        changes[key] = delta
assert not any(k.startswith('eovsam_encoder.') for k in two['model_state_dict'])
weight = Path('/data/hmp/SAMweights/EOVSAM_release_3888_1675_6088_2052_9706.pth')
digest = hashlib.sha256()
with weight.open('rb') as source:
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
assert digest.hexdigest() == 'dd624dc41937d5ff9f13bb81184162d5eb84c165f3f6d4bfea8225d3b07785da'
report = {'steps': [one['global_step'], two['global_step']],
          'losses': [float(one['loss']), float(two['loss'])],
          'updated_parameter_max_abs_deltas': changes,
          'eovsam_excluded_from_checkpoint': True,
          'eovsam_weight_sha256_unchanged': digest.hexdigest(),
          'optimizer_state_count': len(two['optimizer_state_dict']['state'])}
Path(args.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
