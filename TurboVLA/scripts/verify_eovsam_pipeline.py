"""Run the real LIBERO loader, ADE20K visualization and policy backward check."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from turbovla.data.libero_rlds import LiberoRLDSDataset, vla_collate_fn
from turbovla.models.configuration import TurboVLAConfig, EOVSAMConfig
from turbovla.models.turbovla import build_turbovla


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save_panel(raw, resized, result, metadata, path, title):
    raw = raw.permute(1, 2, 0).cpu().numpy()
    prediction = result['sem_seg'].argmax(0).cpu().numpy()
    colors = np.array(metadata.stuff_colors, dtype=np.uint8)
    overlay = np.asarray(0.55 * raw + 0.45 * colors[prediction], dtype=np.uint8)
    Image.fromarray(raw).save(path.with_name(path.stem + '_rgb.png'))
    Image.fromarray(resized['image'].permute(1, 2, 0).numpy().astype(np.uint8)).save(path.with_name(path.stem + '_eovsam_input.png'))
    Image.fromarray(prediction.astype(np.uint8)).save(path.with_name(path.stem + '_labels.png'))
    canvas = Image.new('RGB', (1000, 440), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), title, fill='black')
    draw.text((15, 35), f"Raw {raw.shape[1]}x{raw.shape[0]} RGB", fill='black')
    size = tuple(resized['image'].shape[-2:])
    draw.text((340, 35), f"EOVSAM test resize {size[1]}x{size[0]}", fill='black')
    draw.text((670, 35), 'ADE20K segmentation (restored to raw size)', fill='black')
    canvas.paste(Image.fromarray(raw).resize((300, 300)), (15, 60))
    canvas.paste(Image.fromarray(resized['image'].permute(1, 2, 0).numpy().astype(np.uint8)).resize((300, 300)), (340, 60))
    canvas.paste(Image.fromarray(overlay).resize((300, 300)), (670, 60))
    ids, counts = np.unique(prediction, return_counts=True)
    order = np.argsort(-counts)[:8]
    for j, index in enumerate(order):
        label = int(ids[index])
        x = 15 + (j % 4) * 245
        y = 380 + (j // 4) * 26
        draw.rectangle((x, y, x + 14, y + 14), fill=tuple(colors[label]))
        draw.text((x + 20, y), metadata.stuff_classes[label].split(',')[0] + f' {counts[index] / prediction.size:.0%}', fill='black')
    canvas.save(path)
    return {'input_hw': list(size), 'raw_hw': list(raw.shape[:2]),
            'classes': {metadata.stuff_classes[int(ids[i])]: int(counts[i]) for i in order}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='../outputs/eovsam_verification')
    parser.add_argument('--suites', default='libero_10_no_noops,libero_object_no_noops,libero_spatial_no_noops')
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = TurboVLAConfig(eovsam=EOVSAMConfig(enabled=True))
    config.vision.model_name_or_path = '/data/hmp/Robotics/TurboVLA_Pretrained/dinov3-vitb16-pretrain-lvd1689m'
    config.text.model_name_or_path = '/data/hmp/Robotics/TurboVLA_Pretrained/bert-base-uncased'
    before = digest(config.eovsam.weights_path)
    model = build_turbovla(config).cuda()
    model.train()
    report = {'weight_sha256_before': before, 'model_config': config.to_dict(), 'examples': []}
    first = None
    for suite in args.suites.split(','):
        dataset = LiberoRLDSDataset(
            f'/data/hmp/Robotics/TurboVLA/data/libero/{suite}/1.0.0',
            config.vision.model_name_or_path, chunk_size=config.action.horizon,
            shuffle_buffer=1, step_mix_buffer_size=0,
        )
        batch = vla_collate_fn([next(iter(dataset))])
        samples, instructions, state, actions, valid = batch
        if first is None:
            first = batch
        rgb = samples['rgb'].flatten(0, 1).cuda()
        start = time.time()
        features, results = model.eovsam_encoder.extract(rgb, segment=True)
        torch.cuda.synchronize()
        assert features.shape == (2, 200, 1792) and not features.requires_grad
        assert torch.isfinite(features).all()
        inputs = model.eovsam_encoder.prepare_inputs(rgb)
        item = {'suite': suite, 'instruction': instructions[0], 'features_shape': list(features.shape),
                'feature_mean': features.mean().item(), 'feature_std': features.std().item(),
                'elapsed_seconds': time.time() - start, 'views': []}
        for i, name in enumerate(['agentview', 'wrist']):
            path = output / f'{suite}_{name}.png'
            item['views'].append(save_panel(rgb[i], inputs[i], results[i], model.eovsam_encoder.metadata, path, f'{suite} / {name}'))
        report['examples'].append(item)
        if len(report['examples']) == 1:
            feature_only = model.eovsam_encoder(rgb.reshape(1, 2, 3, 256, 256)).reshape(2, 200, 1792)
            delta = (features - feature_only).abs().max().item()
            assert torch.allclose(features, feature_only, rtol=1e-5, atol=1e-5)
            report['segmentation_vs_feature_only_max_error'] = delta
        del results, features, inputs
        (output / 'verification.json').write_text(json.dumps(report, indent=2))

    samples, instructions, state, actions, valid = first
    samples = {k: v.cuda() for k, v in samples.items()}
    prediction = model(instructions, samples, state.cuda())
    assert prediction.shape == (1, 12, 7) and torch.isfinite(prediction).all()
    loss = ((prediction - actions.cuda()).square() * valid.cuda()[..., None]).sum() / (valid.sum().cuda() * 7)
    loss.backward()
    eovsam = model.eovsam_encoder.model
    assert not eovsam.training and all(not p.requires_grad and p.grad is None for p in eovsam.parameters())
    for layer in model.action_head.decoder.decoder.layers:
        assert layer.object_attn.in_proj_weight.grad is not None
        assert torch.isfinite(layer.object_attn.in_proj_weight.grad).all()
        assert layer.object_attn.in_proj_weight.grad.abs().sum() > 0
    after = digest(config.eovsam.weights_path)
    assert before == after
    report.update({'action_shape': list(prediction.shape), 'backward_loss': loss.item(),
                   'eovsam_frozen_no_grad_eval': True, 'new_attention_gradients_all_layers': True,
                   'checkpoint_has_external_eovsam_weights': any(k.startswith('eovsam_encoder.') for k in model.state_dict()),
                   'weight_sha256_after': after, 'peak_cuda_memory_gib': torch.cuda.max_memory_allocated() / 1024**3})
    (output / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
