# Farm Table OCR

A layout-aware OCR pipeline for digitizing handwritten farm records stored in bordered tabular formats.

## Project Goal

This project aims to convert scanned or photographed farm record sheets into structured CSV/JSON data.

Unlike traditional OCR pipelines that process entire pages directly, this system performs:

1. Table structure recognition
2. Cell segmentation
3. Cell-wise OCR
4. Rule-based postprocessing
5. Structured data export

The project focuses on classical computer vision and image-processing techniques rather than deep-learning-based table understanding.

---

# Motivation

Direct OCR on handwritten farm logs performs poorly because:

- table borders interfere with text recognition,
- handwritten text is noisy and inconsistent,
- spatial structure is important,
- OCR engines are not inherently table-aware.

This project explores a structure-first workflow where the table layout is reconstructed before OCR is applied.

---

# Proposed Pipeline

Input Image
→ Preprocessing
→ Deskew / Perspective Correction
→ Table Detection
→ Horizontal & Vertical Border Extraction
→ Grid Reconstruction
→ Cell Segmentation
→ Border Removal
→ Local OCR
→ Postprocessing & Validation
→ CSV/JSON Export

---

# Core Techniques

## Image Preprocessing
- grayscale conversion
- adaptive thresholding
- Sauvola / Otsu binarization
- noise reduction
- deskewing

## Table Structure Recognition
- morphological line extraction
- horizontal/vertical kernel operations
- contour analysis
- line intersection analysis
- coordinate clustering

## OCR
- cell-wise OCR
- handwritten text recognition
- field-aware parsing

## Postprocessing
- date correction
- numeric validation
- vocabulary-based correction
- confidence scoring

---

# Current Scope

The initial implementation assumes:

- fully bordered tables,
- one primary table per page,
- minimal merged cells,
- scanned or near-front-facing images.

Support for more complex layouts will be added incrementally.

---

# Research Inspiration

This project is inspired by classical document-analysis and table-recognition research, including:

- morphology-based table detection,
- non-learning table structure reconstruction,
- adaptive document binarization techniques.

---

# Planned Milestones

## Phase 1
- preprocessing
- table detection
- cell extraction

## Phase 2
- local OCR integration
- CSV reconstruction

## Phase 3
- rule-based correction
- validation system

## Phase 4
- robustness improvements
- merged-cell handling
- multiple tables
- noisy image handling

---

# Tech Stack

- Python
- OpenCV
- NumPy
- Tesseract / PaddleOCR / TrOCR
- Pandas

---

# Expected Output

The final system should produce:

- structured CSV exports,
- OCR confidence metadata,
- cell-level positional mapping,
- optional manual-review flags.

---

# Status

Prospective research and implementation project.