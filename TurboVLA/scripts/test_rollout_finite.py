"""Run the public rollout entry while checking actions before sanitization."""
import json
import os
from pathlib import Path
import numpy as np
from turbovla.evaluation import policy
from vla_adapter.rollout import main

original = policy.sanitize_pred_chunk
report = {'calls': 0, 'all_raw_actions_finite': True, 'shapes': []}


def checked(prediction):
    assert np.isfinite(prediction).all(), 'Raw action output contains NaN/Inf'
    assert prediction.shape == (12, 7), prediction.shape
    report['calls'] += 1
    shape = list(prediction.shape)
    if shape not in report['shapes']:
        report['shapes'].append(shape)
    return original(prediction)


policy.sanitize_pred_chunk = checked
main()
assert report['calls'] > 0
Path(os.environ['FINITE_REPORT_PATH']).write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
