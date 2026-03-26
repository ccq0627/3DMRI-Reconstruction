import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import yaml

sys.path.append("./")
from r2_gaussian.utils.image_utils import psnr
from r2_gaussian.utils.loss_utils import ssim

THRESHOLD = 0.01  # Minimum mean absolute value of GT slice to consider for evaluation

def _load_volume(path: str) -> torch.Tensor:
	"""Load a 3D volume from .npy or .pt/.pth file into float32 tensor."""
	ext = os.path.splitext(path)[1].lower()
	if ext == ".npy":
		arr = np.load(path)
		vol = torch.from_numpy(arr)
	elif ext in {".pt", ".pth"}:
		data = torch.load(path, map_location="cpu")
		if isinstance(data, torch.Tensor):
			vol = data
		elif isinstance(data, dict) and "vol" in data:
			vol = data["vol"]
		else:
			raise ValueError(f"Unsupported tensor content in: {path}")
	else:
		raise ValueError(f"Unsupported file extension: {ext}. Use .npy/.pt/.pth")

	if vol.ndim != 3:
		raise ValueError(f"Expected 3D volume [D, H, W], got shape {tuple(vol.shape)}")
	return vol.float()


@torch.no_grad()
def evaluate_slices(
	vol_gt: torch.Tensor,
	vol_pred: torch.Tensor,
	pixel_max: float,
	min_tissue: float,
) -> Dict:
	if vol_gt.shape != vol_pred.shape:
		raise ValueError(
			f"Shape mismatch: gt {tuple(vol_gt.shape)} vs pred {tuple(vol_pred.shape)}"
		)

	n_slices = vol_gt.shape[0]
	per_slice: List[Dict[str, float]] = []
	skipped_slices: List[int] = []
	psnr_values: List[float] = []
	ssim_values: List[float] = []
	pred_max = float(vol_pred.max())
	if pred_max > 0:
		vol_pred_eval = vol_pred / pred_max  # Normalize pred to [0, 1] for fair PSNR/SSIM
	else:
		vol_pred_eval = vol_pred.clone()

	for i in range(n_slices):

		gt_slice = vol_gt[i]
		if float(gt_slice.abs().mean()) < min_tissue:
			skipped_slices.append(int(i))
			continue

		pred_slice = vol_pred_eval[i]

		gt_4d = gt_slice[None, None]
		pred_4d = pred_slice[None, None]

		psnr_i = psnr(gt_4d, pred_4d, pixel_max=pixel_max).item()
		ssim_i = ssim(gt_4d, pred_4d).item()

		per_slice.append(
			{
				"slice_index": int(i),
				"psnr": float(psnr_i),
				"ssim": float(ssim_i),
			}
		)
		psnr_values.append(float(psnr_i))
		ssim_values.append(float(ssim_i))

	mean_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
	mean_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0

	return {
		"num_slices": int(n_slices),
		"num_valid_slices": int(len(per_slice)),
		"num_skipped_slices": int(len(skipped_slices)),
		"skipped_slice_indices": skipped_slices,
		"min_tissue": float(min_tissue),
		"per_slice": per_slice,
		"mean": {
			"psnr": mean_psnr,
			"ssim": mean_ssim,
		},
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Evaluate per-slice PSNR/SSIM for reconstructed MRI volume."
	)
	parser.add_argument("--gt", required=True, help="Path to GT volume (.npy/.pt/.pth)")
	parser.add_argument(
		"--pred", required=True, help="Path to reconstructed volume (.npy/.pt/.pth)"
	)
	parser.add_argument(
		"--output",
		default="eval.yaml",
		help="Output yaml path (default: eval.yaml)",
	)
	parser.add_argument(
		"--pixel-max",
		type=float,
		default=None,
		help="Pixel max for PSNR. Default: max value of GT volume.",
	)
	parser.add_argument(
		"--min-tissue",
		type=float,
		default=THRESHOLD,
		help="Skip slice if mean(abs(gt_slice)) < this value.",
	)
	args = parser.parse_args()

	vol_gt = _load_volume(args.gt)
	vol_pred = _load_volume(args.pred)

	pixel_max = args.pixel_max if args.pixel_max is not None else float(vol_gt.max())
	if pixel_max <= 0:
		raise ValueError("pixel_max must be > 0.")

	result = evaluate_slices(
		vol_gt,
		vol_pred,
		pixel_max=pixel_max,
		min_tissue=args.min_tissue,
	)
	result["gt_path"] = args.gt
	result["pred_path"] = args.pred
	result["pixel_max"] = float(pixel_max)

	output_dir = os.path.dirname(args.output)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	with open(args.output, "w", encoding="utf-8") as f:
		yaml.safe_dump(result, f, sort_keys=False, allow_unicode=False)

	print(f"Saved evaluation to: {args.output}")
	print(
		f"Mean metrics -> PSNR: {result['mean']['psnr']:.4f}, SSIM: {result['mean']['ssim']:.4f}"
	)
	print(
		f"Evaluated slices: {result['num_valid_slices']}/{result['num_slices']} "
		f"(skipped: {result['num_skipped_slices']}, min_tissue={args.min_tissue})"
	)


if __name__ == "__main__":
	main()




