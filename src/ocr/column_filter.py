from __future__ import annotations

# Header names listed here are removed from OCR output after the header row is read.
# Matching is case-insensitive and ignores repeated surrounding/internal whitespace.
FILTER_OUT_COLUMNS: set[str] = set()


# This filter is overridden by the filter list passed in streamlit app UI.
