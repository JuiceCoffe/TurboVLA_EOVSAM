"""Check strict checkpoint loading and train/inference/train with real LIBERO RGB."""
import argparse
import json
from pathlib import Path
import torch
from turbovla.data.libero_rlds import LiberoRLDSDataset, vla_collate_fn
from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.turbovla import build_turbovla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = TurboVLAConfig.from_mapping(checkpoint['model_config'])
    model = build_turbovla(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    model.cuda()
    data = LiberoRLDSDataset(
        '/data/hmp/Robotics/TurboVLA/data/libero/libero_spatial_no_noops/1.0.0',
        config.vision.model_name_or_path, chunk_size=config.action.horizon,
        shuffle_buffer=1, step_mix_buffer_size=0,
    )
    samples, text, states, actions, valid = vla_collate_fn([next(iter(data))])
    samples = {k: v.cuda() for k, v in samples.items()}
    states, actions, valid = states.cuda(), actions.cuda(), valid.cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    report = {'checkpoint': args.checkpoint, 'strict_load': True, 'phases': []}
    for phase in ('train_before', 'inference', 'train_after'):
        model.train(phase != 'inference')
        if phase == 'inference':
            with torch.inference_mode():
                prediction = model(text, samples, states)
            assert not prediction.requires_grad
            item = {'phase': phase, 'finite': bool(torch.isfinite(prediction).all()),
                    'action_shape': list(prediction.shape)}
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(text, samples, states)
                loss = ((prediction.float() - actions).abs() * valid[..., None]).sum() / (valid.sum() * actions.shape[-1])
            assert torch.isfinite(loss)
            loss.backward()
            gradients = {}
            for i, projector in enumerate(model.action_head.decoder.object_projections):
                grad = projector.weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
                gradients[str(i)] = grad.norm().item()
            for layer in model.action_head.decoder.decoder.layers:
                grad = layer.object_attn.in_proj_weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            item = {'phase': phase, 'loss': loss.item(), 'projection_gradient_norms': gradients}
        external = model.eovsam_encoder.model
        assert not external.training
        assert all(not p.requires_grad and p.grad is None for p in external.parameters())
        item['eovsam_frozen_eval_no_grad'] = True
        report['phases'].append(item)
        print(json.dumps(item), flush=True)
    assert not any(k.startswith('eovsam_encoder.') for k in model.state_dict())
    report['eovsam_excluded_from_checkpoint'] = True
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
