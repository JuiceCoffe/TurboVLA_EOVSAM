import argparse
import json
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('run_id')
args = parser.parse_args()
api = wandb.Api()
run = api.run(f'{api.default_entity}/TurboVLA-EOVSAM-LIBERO/{args.run_id}')
keys = ('world_size', 'batch_size', 'grad_accum_steps', 'effective_batch_size_global',
        'max_steps', 'warmup_steps', 'precision', 'dinov3_precision', 'eval_steps',
        'eval_num_trials_per_task', 'seed', 'lr', 'head_lr', 'dinov3_lr')
keys += ('use_eovsam', 'eovsam_train_input_size', 'eval_cuda_visible_devices')
print(json.dumps({'url': run.url, 'state': run.state,
                  'config': {k: run.config.get(k) for k in keys},
                  'summary': {k: v for k, v in dict(run.summary).items()
                              if k.startswith(('train/', 'libero_success_rate/', 'eval/'))}}, indent=2))
