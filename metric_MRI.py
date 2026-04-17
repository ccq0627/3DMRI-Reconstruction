import argparse
import os
import sys
from typing import Dict, List
import cv2
import numpy as np
import torch
import yaml

sys.path.append("./")
from r2_gaussian.utils.image_utils import psnr
from r2_gaussian.utils.loss_utils import ssim
from lpipsPyTorch import lpips
THRESHOLD = 0.01  # Minimum mean absolute value of GT slice to consider for evaluation

def get_outer_mask(img: torch.Tensor) -> np.ndarray:
	img_slice = img.detach().cpu().numpy()
	img_u8 = np.ascontiguousarray(np.clip(img_slice * 255.0, 0, 255).astype(np.uint8))

	edges = cv2.Canny(img_u8, 50, 150)

	# 2. 形态学闭运算连接边缘
	kernel = np.ones((5,5), np.uint8)
	closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

	# 3. 膨胀增强连接
	dilated = cv2.dilate(closed, kernel, iterations=2)

	# 4. 填充轮廓内部
	contours, hierarchy = cv2.findContours(
		dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
	)

	# 5. 创建mask并填充
	mask = np.zeros(img_u8.shape, dtype=np.uint8)

	# 找到最大轮廓
	if contours:
		largest_contour = max(contours, key=cv2.contourArea)
		cv2.drawContours(mask, [largest_contour], -1, 255, -1)

	# 可选：形态学平滑
	mask = np.ascontiguousarray(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel))
	mask = np.ascontiguousarray(cv2.medianBlur(mask, 5))
	mask01 = mask / 255  # Normalize to [0, 1]
	return mask01


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
	use_mask: bool = True,
	compute_lpips: bool = False,
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
	lpips_values: List[float] = []
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

		if use_mask:
			mask = get_outer_mask(gt_slice)
			mask_t = torch.from_numpy(mask).to(device=gt_slice.device, dtype=gt_slice.dtype)
			gt_slice = gt_slice * mask_t
			pred_slice = pred_slice * mask_t


		gt_4d = gt_slice[None, None]  # [1, 1, H, W] for metric functions
		pred_4d = pred_slice[None, None]

		psnr_i = psnr(gt_4d, pred_4d, pixel_max=pixel_max).item()
		ssim_i = ssim(gt_4d, pred_4d).item()
		lpips_i = lpips(gt_4d, pred_4d, net_type='vgg').item() if compute_lpips else 0.0

		per_slice.append(
			{
				"slice_index": int(i),
				"psnr": float(psnr_i),
				"ssim": float(ssim_i),
				"lpips": float(lpips_i),
			}
		)
		psnr_values.append(float(psnr_i))
		ssim_values.append(float(ssim_i))
		lpips_values.append(float(lpips_i))
		
	mean_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
	mean_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0
	mean_lpips = float(np.mean(lpips_values)) if lpips_values else 0.0

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
			"lpips": mean_lpips,
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
	parser.add_argument("--use_mask", action="store_true", default=True, help="Whether to apply a mask to the slices before evaluation.")
	parser.add_argument(
		"--device",
		type=str,
		default="cpu",
		help="Device to use for evaluation.",
	)
	parser.add_argument("--lpips", action="store_true", help="Whether to compute LPIPS metric (requires GPU).")
	args = parser.parse_args()

	if args.device == "cuda" and torch.cuda.is_available():
		vol_gt = _load_volume(args.gt).cuda()
		vol_pred = _load_volume(args.pred).cuda()
	else:
		vol_gt = _load_volume(args.gt)
		vol_pred = _load_volume(args.pred)

	pixel_max = args.pixel_max if args.pixel_max is not None else float(vol_gt.max())
	lpips = None
	if args.lpips:
		lpips = True
	if pixel_max <= 0:
		raise ValueError("pixel_max must be > 0.")

	result = evaluate_slices(
		vol_gt,
		vol_pred,
		pixel_max=pixel_max,
		min_tissue=args.min_tissue,
		use_mask=args.use_mask,
		compute_lpips=lpips,
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




