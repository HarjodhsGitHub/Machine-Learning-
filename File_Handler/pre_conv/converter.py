import os
import sys
import argparse
from pathlib import Path

try:
	from PIL import Image
except ImportError:
	Image = None


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def list_input_files(input_path: Path, exts: set[str]) -> list[Path]:
	if input_path.is_dir():
		return [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
	if input_path.is_file() and input_path.suffix.lower() in exts:
		return [input_path]
	return []


def ensure_output_dir(path: Path) -> None:
	if path.suffix:  # looks like a file path
		path.parent.mkdir(parents=True, exist_ok=True)
	else:
		path.mkdir(parents=True, exist_ok=True)


def convert_image_to_pdf(images: list[Path], output: Path, merge: bool = False) -> None:
	if Image is None:
		raise RuntimeError("Pillow (PIL) is required. Install with: pip install pillow")

	if not images:
		print("No input images found matching supported types.")
		return

	if merge:
		ensure_output_dir(output)
		# If output is a directory, default to merged.pdf
		out_file = output if output.suffix.lower() == ".pdf" else (output / "merged.pdf")
		pdf_pages = []
		for img_path in images:
			with Image.open(img_path) as im:
				im_converted = im.convert("RGB")
				pdf_pages.append(im_converted)
		first, rest = pdf_pages[0], pdf_pages[1:]
		first.save(out_file, save_all=True, append_images=rest)
		print(f"Created: {out_file}")
		return

	# Per-file PDF outputs
	for img_path in images:
		with Image.open(img_path) as im:
			rgb = im.convert("RGB")
			if output.suffix.lower() == ".pdf":
				# If a file path was provided, place alongside it with unique name
				out_dir = output.parent
			else:
				out_dir = output
			ensure_output_dir(out_dir)
			out_file = out_dir / (img_path.stem + ".pdf")
			rgb.save(out_file)
			print(f"Created: {out_file}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Flexible file converter. Currently supports image -> PDF."
	)
	parser.add_argument(
		"input",
		help="Input file or directory (defaults to File_Handler/pre_conv if omitted)",
		type=Path,
		nargs="?",
		default=None,
	)
	parser.add_argument(
		"--input-type",
		choices=["image"],
		default="image",
		help="Type of input files (currently only 'image').",
	)
	parser.add_argument(
		"--output-type",
		choices=["pdf"],
		default="pdf",
		help="Desired output type (currently only 'pdf').",
	)
	parser.add_argument(
		"--output",
		type=Path,
		required=False,
		default=None,
		help="Output file or directory. Defaults to File_Handler/post_conv.",
	)
	parser.add_argument(
		"--merge",
		action="store_true",
		help="Merge all input images into a single PDF.",
	)
	return parser


def main(argv: list[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)

	# Resolve workspace-relative default paths for pre/post conversion
	workspace_root = Path(__file__).resolve().parents[2]
	default_pre = workspace_root / "File_Handler" / "pre_conv"
	default_post = workspace_root / "File_Handler" / "post_conv"

	# Default input to pre_conv directory if not provided
	if args.input is None:
		args.input = default_pre

	# Default output to post_conv directory if not provided
	if args.output is None:
		args.output = default_post

	if args.input_type == "image" and args.output_type == "pdf":
		images = list_input_files(args.input, SUPPORTED_IMAGE_EXTS)
		convert_image_to_pdf(images, args.output, merge=args.merge)
		return 0

	print("Unsupported conversion pathway.")
	return 1


if __name__ == "__main__":
	sys.exit(main())

