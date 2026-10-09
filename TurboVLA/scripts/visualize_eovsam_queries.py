"""Save every EOVSAM decoder query over a real LIBERO RGB sample."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from turbovla.data.libero_rlds import LiberoRLDSDataset, vla_collate_fn
from turbovla.models.configuration import EOVSAMConfig
from turbovla.models.eovsam_encoder import FrozenEOVSAMEncoder


def render(raw, mask_logits, box, rank, query_id, score, void_similarity, path):
    rgb = raw.permute(1, 2, 0).cpu().numpy()
    base = Image.fromarray((rgb.astype(np.float32) * 0.38).astype(np.uint8), 'RGB')
    mask = (mask_logits.sigmoid().cpu().numpy() > 0.5)
    overlay = np.asarray(base).copy()
    overlay[mask] = (0.32 * overlay[mask] + 0.68 * np.array([30, 235, 195])).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = [float(x) for x in box]
    width, height = image.size
    rect = (max(0, min(width - 1, x0)), max(0, min(height - 1, y0)),
            max(0, min(width - 1, x1)), max(0, min(height - 1, y1)))
    if rect[2] > rect[0] and rect[3] > rect[1]:
        draw.rectangle(rect, outline=(255, 190, 30), width=2)
    label = f'#{rank:03d} query {query_id:03d}  foreground {score:.4f}  void cosine {void_similarity:.3f}'
    canvas = Image.new('RGB', (max(width * 2, 512), height * 2 + 31), (24, 24, 24))
    canvas.paste(image.resize((width * 2, height * 2), Image.Resampling.NEAREST), (0, 31))
    ImageDraw.Draw(canvas).text((7, 8), label, fill='white', font=ImageFont.load_default())
    canvas.save(path)
    return int(mask.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suites', default='libero_10_no_noops,libero_object_no_noops,libero_spatial_no_noops')
    parser.add_argument('--output_dir', default='../outputs/eovsam_query_visualization')
    parser.add_argument('--weights', default=EOVSAMConfig.weights_path)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    encoder = FrozenEOVSAMEncoder(EOVSAMConfig(enabled=True, weights_path=args.weights)).cuda().eval()
    report = {'score': 'foreground_score = 1 - softmax(raw SAM class logits including void)[void]',
              'ranking': 'ascending void_similarity (least void-like first)',
              'void_similarity': 'cosine of normalized pixel-enriched query and normalized void_embedding; smaller means more foreground',
              'void_logit': 'void_similarity multiplied by the learned logit scale',
              'mask_threshold': 0.5, 'views': []}
    for suite in args.suites.split(','):
        dataset = LiberoRLDSDataset(
            f'/data/hmp/Robotics/TurboVLA/data/libero/{suite}/1.0.0',
            '/data/hmp/Robotics/TurboVLA_Pretrained/dinov3-vitb16-pretrain-lvd1689m',
            chunk_size=12, shuffle_buffer=1, step_mix_buffer_size=0,
        )
        samples, instructions, *_ = vla_collate_fn([next(iter(dataset))])
        rgb = samples['rgb'].flatten(0, 1).cuda()
        features, predictions = encoder.extract(rgb, query_predictions=True)
        assert features.shape[:2] == (2, 200) and len(predictions) == 2
        for view, (raw, pred) in enumerate(zip(rgb, predictions)):
            name = ['agentview', 'wrist'][view]
            folder = output / f'{suite}_{name}'
            folder.mkdir(parents=True, exist_ok=True)
            for previous in folder.glob('[0-9][0-9][0-9]_query_[0-9][0-9][0-9].png'):
                previous.unlink()
            Image.fromarray(raw.permute(1, 2, 0).cpu().numpy()).save(folder / 'original.png')
            scores = pred['foreground_score'].cpu()
            void_logits = pred['void_logit'].cpu()
            void_similarities = pred['void_similarity'].cpu()
            boxes = pred['boxes_xyxy'].cpu()
            masks = pred['mask_logits'].cpu()
            # Lower cosine similarity to the void embedding means less void-like.
            order = torch.argsort(void_similarities, descending=False, stable=True).tolist()
            records = []
            for rank, query_id in enumerate(order, 1):
                filename = f'{rank:03d}_query_{query_id:03d}.png'
                pixels = render(raw, masks[query_id], boxes[query_id], rank, query_id,
                                scores[query_id].item(), void_similarities[query_id].item(), folder / filename)
                records.append({'rank': rank, 'query_id': query_id, 'foreground_score': scores[query_id].item(),
                                'void_similarity': void_similarities[query_id].item(),
                                'void_logit': void_logits[query_id].item(), 'box_xyxy': boxes[query_id].tolist(),
                                'mask_pixels': pixels, 'file': filename})
            (folder / 'queries.json').write_text(json.dumps(records, indent=2))
            report['views'].append({'suite': suite, 'view': name, 'instruction': instructions[0],
                                    'folder': str(folder), 'queries': len(records),
                                    'feature_shape': list(features[view].shape)})
            print(f'Saved {len(records)} queries to {folder}', flush=True)
        (output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
