# Pre-Conversion File Converter

This utility provides a simple CLI to convert images to PDF.

## Requirements
- Python 3.10+
- Pillow (`pip install pillow`)

## Usage

Convert all images in `File_Handler/pre_conv` to PDFs in `File_Handler/post_conv` (defaults):

```bash
python File_Handler/pre_conv/converter.py --input-type image --output-type pdf
```

Convert all images in a specific folder to individual PDFs:

```bash
python File_Handler/pre_conv/converter.py File_Handler/pre_conv/sample_images --input-type image --output-type pdf
```

Merge multiple images into a single PDF (written to `File_Handler/post_conv/merged.pdf` by default):

```bash
python File_Handler/pre_conv/converter.py File_Handler/pre_conv/sample_images --input-type image --output-type pdf --merge
```

Convert a single image to PDF:

```bash
python File_Handler/pre_conv/converter.py File_Handler/pre_conv/sample_images/example.jpg --output File_Handler/post_conv
```

- If `--output` points to a directory, PDFs are written inside it.
- If `--output` is omitted, files are written to `File_Handler/post_conv`.
- If `--merge` is used without a filename, output defaults to `File_Handler/post_conv/merged.pdf`.
