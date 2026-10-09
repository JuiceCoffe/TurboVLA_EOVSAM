from pathlib import Path
import json
import tensorflow as tf
root = Path('/data/hmp/Robotics/TurboVLA/data/libero')
report = {}
for folder in sorted(root.glob('*/1.0.0')):
    count = episodes = 0
    for shard in sorted(folder.glob('*.tfrecord-*')):
        for record in tf.compat.v1.io.tf_record_iterator(str(shard)):
            example = tf.train.Example.FromString(record)
            features = example.features.feature
            if 'steps/observation/image' not in features:
                raise RuntimeError(list(features.keys()))
            count += len(features['steps/observation/image'].bytes_list.value)
            episodes += 1
    report[folder.parent.name] = {'steps': count, 'episodes': episodes}
    print(folder.parent.name, report[folder.parent.name], flush=True)
out = Path('/data/hmp/Robotics/TurboVLA_EOVSAM/outputs/highres_cost_estimate')
out.mkdir(parents=True, exist_ok=True)
(out / 'frame_counts.json').write_text(json.dumps(report, indent=2))
