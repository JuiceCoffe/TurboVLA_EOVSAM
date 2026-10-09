import os
import sys
import io
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
os.environ.setdefault('LIBERO_CONFIG_PATH', '/data/hmp/Robotics/TurboVLA_libero_config')
sys.path.insert(0, '/data/hmp/Robotics/LIBERO')
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

out = Path('/data/hmp/Robotics/TurboVLA_EOVSAM/outputs/highres_cost_estimate')
out.mkdir(parents=True, exist_ok=True)
report = {'hardware': 'RTX 3090; idle at test start', 'iterations_per_mode': 30, 'scenes': []}
def render_pair(env, size):
    return [env.sim.render(width=size, height=size, camera_name=camera)
            for camera in ('agentview', 'robot0_eye_in_hand')]
def measure(call):
    for _ in range(5):
        call()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        call()
        times.append((time.perf_counter() - start) * 1000)
    return {'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)),
            'mean_ms': float(np.mean(times))}
for suite_name in ('libero_10', 'libero_object', 'libero_spatial'):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    for task_id in (0, 1):
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file),
            camera_heights=256, camera_widths=256,
        )
        try:
            env.seed(0)
            env.reset()
            env.set_init_state(suite.get_task_init_states(task_id)[0])
            scene = {'suite': suite_name, 'task': task.name}
            scene['render_two_256'] = measure(lambda: render_pair(env, 256))
            scene['render_two_1152'] = measure(lambda: render_pair(env, 1152))
            scene['render_two_256_plus_two_1152'] = measure(lambda: (render_pair(env, 256), render_pair(env, 1152)))
            scene['extra_pair_median_ms'] = scene['render_two_256_plus_two_1152']['median_ms'] - scene['render_two_256']['median_ms']
            sizes = []
            for index, image in enumerate(render_pair(env, 1152)):
                entry = {'camera': ['agentview', 'wrist'][index], 'raw_bytes': int(image.nbytes)}
                pil = Image.fromarray(image[::-1, ::-1].copy())
                for format, options, label in [('PNG', {}, 'png'), ('JPEG', {'quality': 95, 'subsampling': 0}, 'jpeg95_444')]:
                    buffer = io.BytesIO()
                    start = time.perf_counter()
                    pil.save(buffer, format=format, **options)
                    entry[label + '_bytes'] = buffer.tell()
                    entry[label + '_encode_ms'] = (time.perf_counter() - start) * 1000
                sizes.append(entry)
            scene['image_sizes'] = sizes
            report['scenes'].append(scene)
            (out / 'benchmark.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(scene), flush=True)
        finally:
            env.close()
