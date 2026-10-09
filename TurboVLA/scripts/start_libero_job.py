"""Detach the authorized two-GPU training job from the SSH session."""
import json
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[1]
mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
if mode not in ('full', 'smoke', 'capacity'):
    raise ValueError(mode)
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 8
folder = repo.parent / 'outputs' / f'libero_ddp_{mode}'
folder.mkdir(parents=True, exist_ok=True)
with (folder / 'launch.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(
        ['bash', 'scripts/launch_libero_ddp.sh', mode, str(batch)], cwd=repo,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
metadata = {'pid': process.pid, 'mode': mode, 'batch_size_per_gpu': batch,
            'grad_accum_steps': 64 // batch, 'effective_batch_size': 128,
            'log': str(folder / 'launch.log')}
(folder / 'launcher.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata))
