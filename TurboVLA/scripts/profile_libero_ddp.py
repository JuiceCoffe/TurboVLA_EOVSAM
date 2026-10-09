"""Run the real DDP trainer with one-time EOVSAM data-flow and memory checks."""
import json
import os
from pathlib import Path
import torch
from turbovla.training import train_mixed, trainer
from turbovla.models.eovsam_encoder import FrozenEOVSAMEncoder

rank = int(os.environ.get('RANK', '0'))
folder = Path('../outputs/libero_ddp_capacity')
folder.mkdir(parents=True, exist_ok=True)
report = {'rank': rank, 'events': {}}
original = trainer.build_model_architecture
original_prepare = FrozenEOVSAMEncoder.prepare_inputs


def checked_prepare(self, rgb, **kwargs):
    inputs = original_prepare(self, rgb, **kwargs)
    if kwargs.get('use_training_resolution'):
        assert all(tuple(x['image'].shape[-2:]) == (512, 512) for x in inputs)
    if 'eovsam_input_hw' not in report['events']:
        record('eovsam_input_hw', list(inputs[0]['image'].shape[-2:]))
        external = self.model
        original_text = external.detector.backbone.forward_text

        def count_text(*args, **kw):
            report['sam_text_encoder_calls'] = report.get('sam_text_encoder_calls', 0) + 1
            return original_text(*args, **kw)

        def unexpected(*args, **kw):
            raise AssertionError('Feature-only training entered a classification/segmentation output head')

        external.detector.backbone.forward_text = count_text
        external.detector._update_scores_and_boxes = unexpected
        external.detector._run_segmentation_heads = unexpected
        external.detector.compute_open_vocab_classification_scores = unexpected
        external._compute_attn_pool_cls_logits = unexpected
    return inputs


FrozenEOVSAMEncoder.prepare_inputs = checked_prepare


def record(name, value):
    report['events'][name] = value
    (folder / f'feature_flow_rank{rank}.json').write_text(json.dumps(report, indent=2))
    print(f'[FEATURE_FLOW rank={rank}] {name}: {value}', flush=True)


def instrument(args):
    model = original(args)
    captured = {}

    def encoder_hook(module, inputs, output):
        if 'encoder' in report['events']:
            return
        assert output.shape == (args.batch_size, 400, 1792)
        assert not output.requires_grad and torch.isfinite(output).all()
        external = module.model
        assert not external.training and all(not p.requires_grad for p in external.parameters())
        captured['features'] = output.detach().clone()
        record('encoder', {'shape': list(output.shape), 'std': output.float().std().item(),
                           'requires_grad': output.requires_grad, 'external_frozen_eval': True})
        report['sam_text_encoder_calls_after_first_batch'] = report.get('sam_text_encoder_calls', 0)

    def action_hook(module, inputs):
        if 'action_head_input' in report['events']:
            return
        features = inputs[2]
        extracted = captured.pop('features')
        assert torch.equal(features, extracted.to(features))
        record('action_head_input', {'shape': list(features.shape),
                                     'equals_eovsam_output': True})

    def decoder_hook(module, inputs, kwargs):
        if 'object_cross_attention_memory' in report['events']:
            return
        memory = kwargs['object_memory']
        assert memory.shape == (args.batch_size, 400, 256)
        assert torch.isfinite(memory).all() and memory.requires_grad
        record('object_cross_attention_memory', {'shape': list(memory.shape),
                                                 'requires_grad': memory.requires_grad})

    def gradient_hook(name):
        def check(gradient):
            if name not in report['events']:
                assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                record(name, {'gradient_norm': gradient.norm().item()})
        return check

    model.eovsam_encoder.register_forward_hook(encoder_hook)
    model.action_head.register_forward_pre_hook(action_hook)
    model.action_head.decoder.decoder.register_forward_pre_hook(decoder_hook, with_kwargs=True)
    for i, projector in enumerate(model.action_head.decoder.object_projections):
        projector.weight.register_hook(gradient_hook(f'projection_{i}'))
    for i, layer in enumerate(model.action_head.decoder.decoder.layers):
        layer.object_attn.in_proj_weight.register_hook(gradient_hook(f'object_attention_{i}'))
    return model


trainer.build_model_architecture = instrument
train_mixed.main()
report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
report['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
report['device_total_gib'] = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
assert report['peak_reserved_gib'] < report['device_total_gib'] - 2, report
report['safe_memory_margin'] = True
assert report['sam_text_encoder_calls'] == report['sam_text_encoder_calls_after_first_batch']
report['no_text_reencoding_after_warmup'] = True
report['classification_and_segmentation_output_heads_skipped'] = True
(folder / f'feature_flow_rank{rank}.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
