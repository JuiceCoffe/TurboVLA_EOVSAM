import os
import json
from pathlib import Path
import sys
os.environ.setdefault('LIBERO_CONFIG_PATH', '/data/hmp/Robotics/TurboVLA_libero_config')
sys.path.insert(0, '/data/hmp/Robotics/LIBERO')
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

suite = benchmark.get_benchmark_dict()['libero_10']()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=1152, camera_widths=1152)
try:
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    output = Path('/data/hmp/Robotics/TurboVLA_EOVSAM/outputs/native_resolution_probe')
    output.mkdir(exist_ok=True, parents=True)
    report = {'task': task.name, 'resolution_requested': 1152, 'method': 'native simulator camera rendering; no resize'}
    for key in ('agentview_image', 'robot0_eye_in_hand_image'):
        image = obs[key]
        assert image.shape == (1152, 1152, 3)
        Image.fromarray(image[::-1, ::-1].copy()).save(output / f'{key}.png')
        report[key] = list(image.shape)
    (output / 'probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
finally:
    env.close()
