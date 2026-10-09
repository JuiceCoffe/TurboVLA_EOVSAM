"""Validate suite metric names and retention using synthetic result files."""
import json
from pathlib import Path
import sys
import tempfile
from turbovla.training import trainer

sys.argv = ['test', '--dinov3_path', 'unused', '--bert_path', 'unused', '--pretrained_init_ckpt', 'unused']
args = trainer.parse_args()
args.stats_path = 'unused'
args.stats_key = 'unused'
args.normalize_binary_gripper = 'auto'
with tempfile.TemporaryDirectory() as folder:
    args.eval_output_dir = folder
    counts = iter([100, 200, 300, 400])
    original = trainer.subprocess.run

    def mock_evaluate(command, **kwargs):
        path = command[command.index('--result_json_path') + 1]
        count = next(counts)
        Path(path).write_text(json.dumps({'total_episodes': 500, 'total_successes': count,
                                         'overall_success_rate': count / 500, 'tasks': []}))

    trainer.subprocess.run = mock_evaluate
    try:
        metrics = trainer.run_libero_evaluation(args, 'unused.pth', 10000)
    finally:
        trainer.subprocess.run = original
    success = {k: v for k, v in metrics.items() if k.startswith('libero_success_rate/')}
    assert set(success) == {'libero_success_rate/' + key for key in
                            ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'overall')}
    assert success['libero_success_rate/overall'] == 0.5
    assert 'eval/overall_success_rate' not in metrics
    assert all(f'eval/{suite}/success_rate' not in metrics for suite in
               ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'))
    args.checkpoint_dir = folder
    args.keep_last_checkpoints = 2
    args.eval_steps = 20
    args.max_steps = 80
    for step in (10, 20, 30, 40):
        Path(folder, f'turbovla_step_{step}.pth').touch()
    Path(folder, 'unrelated.pth').touch()
    trainer.prune_training_checkpoints(args)
    assert not Path(folder, 'turbovla_step_10.pth').exists()
    assert all(Path(folder, f'turbovla_step_{step}.pth').exists() for step in (20, 30, 40))
    assert Path(folder, 'unrelated.pth').exists()
    print(json.dumps({'metric_section': success, 'checkpoint_retention': 'passed'}, indent=2))
