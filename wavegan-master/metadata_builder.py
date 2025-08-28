#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Metadata builder for vowel length / quality / tone datasets.

Scans a root data directory with expected structure like:
  <root>/<Language>/(long vowels-#VT|short vowels-#VT|...)/<filename>.mp3

Extracted fields per file (best-effort, robust to irregularities):
  filepath: absolute path
  relpath: path relative to root
  language: top-level folder name
  length_class: 'long' or 'short' (derived from containing directory name prefix)
  pattern_tag: trailing token after '-' in directory name (e.g. #VT, #TV, #DV, #VD, #VVT)
  filename: base filename without extension
  ipa_raw: IPA + ancillary symbols (before tone digits / gender / region markers)
  ipa: IPA sequence with length mark 'ː' retained, strips gender/region and tone digits
  tone: trailing digits (e.g. 55, 33, 21, 2, 45) if present; else ''
  gender: 'F' or 'M' if Chinese 女/男 found at end (Vietnamese subset); else ''
  region: region marker inside parentheses e.g. (北),(南) extracted; else ''
  has_colon_len: 1 if 'ː' in ipa, else 0
  vowel_length_dir: length class derived from directory (sanity check vs ipa mark)

Output: CSV (UTF-8) with header.

Usage:
  python metadata_builder.py --data_root ../code_of_miss --output metadata.csv 

The script does not require heavy dependencies; uses only stdlib.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Dict, Iterable, List, Optional

LANG_PATTERN = re.compile(r"^(Cantonese|Thai|Vietnamese)$", re.IGNORECASE)

# Matches tone digits at end of token (1-5 digits, usually 1-2).
TONE_RE = re.compile(r"(?P<ipa>.*?)(?P<tone>\d{1,3})$")

# Region in parentheses, e.g. ɑːm(北)
REGION_RE = re.compile(r"^(?P<stem>.*?)(?:\((?P<region>[^)]+)\))$")

# Gender markers at very end (Chinese chars 女 / 男)
GENDER_RE = re.compile(r"^(?P<stem>.*?)(?P<gender>[男女])$")

CHINESE_GENDER_MAP = {u"女": "F", u"男": "M"}


def iter_audio_files(root: str, exts=(".mp3", ".wav", ".ogg")) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                yield os.path.join(dirpath, fn)


def parse_container_dir(dir_name: str):  # -> Tuple[str, str]
    """Return (length_class, pattern_tag)."""
    name = dir_name.strip()
    length_class = 'long' if name.lower().startswith('long') else ('short' if name.lower().startswith('short') else '')
    # pattern tag after '-' if present
    pattern_tag = ''
    if '-' in name:
        pattern_tag = name.split('-', 1)[1].strip()
    return length_class, pattern_tag


def clean_ipa_token(token: str) -> Dict[str, str]:
    """Extract ipa, tone, gender, region from token (without extension).

    token may include spaces (rare), region in parentheses, gender char, trailing digits (tone).
    Parsing order: separate gender, then region, then tone digits.
    """
    original = token
    gender = ''
    region = ''
    tone = ''

    # Strip spaces inside token early (keep those that separate Chinese char from IPA managed elsewhere).
    token = token.strip()

    # Gender
    m = GENDER_RE.match(token)
    if m:
        token = m.group('stem')
        gender = CHINESE_GENDER_MAP.get(m.group('gender'), '')

    # Region
    m = REGION_RE.match(token)
    if m:
        token = m.group('stem')
        region = m.group('region')

    # Tone digits
    m = TONE_RE.match(token)
    if m:
        token = m.group('ipa')
        tone = m.group('tone')

    ipa = token
    return {
        'ipa_raw': original,
        'ipa': ipa,
        'tone': tone,
        'gender': gender,
        'region': region,
    }


def parse_filename(path: str, data_root: str) -> Optional[Dict[str, str]]:
    relpath = os.path.relpath(path, data_root)
    parts = relpath.split(os.sep)
    if len(parts) < 3:
        # Expect at least Language / length-dir / file
        return None
    language = parts[0]
    if not LANG_PATTERN.match(language):
        # Skip unknown top-level
        return None
    length_class, pattern_tag = parse_container_dir(parts[1])
    base = os.path.splitext(parts[-1])[0]

    # For filenames containing space(s) with Chinese characters + IPA, take last whitespace-separated segment as IPA token
    tokens = base.split()
    ipa_token = tokens[-1] if tokens else base
    meta = clean_ipa_token(ipa_token)

    row = {
        'filepath': os.path.abspath(path),
        'relpath': relpath,
        'language': language,
        'length_class': length_class,
        'pattern_tag': pattern_tag,
        'filename': base,
        'ipa_raw': meta['ipa_raw'],
        'ipa': meta['ipa'],
        'tone': meta['tone'],
        'gender': meta['gender'],
        'region': meta['region'],
        'has_colon_len': '1' if 'ː' in meta['ipa'] else '0',
        'vowel_length_dir': length_class,
    }
    return row


def build_metadata(data_root: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for fp in iter_audio_files(data_root):
        row = parse_filename(fp, data_root)
        if row:
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, str]], output_path: str) -> None:
    if not rows:
        print("No rows to write", file=sys.stderr)
        return
    fieldnames = [
        'filepath', 'relpath', 'language', 'length_class', 'pattern_tag',
        'filename', 'ipa_raw', 'ipa', 'tone', 'gender', 'region',
        'has_colon_len', 'vowel_length_dir'
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {output_path}")


def summarize(rows: List[Dict[str, str]]) -> None:
    from collections import Counter
    lang_counts = Counter(r['language'] for r in rows)
    length_counts = Counter(r['length_class'] for r in rows)
    pattern_counts = Counter(r['pattern_tag'] for r in rows)
    print("Languages:")
    for k, v in lang_counts.items():
        print(f"  {k}: {v}")
    print("Length classes:")
    for k, v in length_counts.items():
        print(f"  {k}: {v}")
    print("Pattern tags:")
    for k, v in pattern_counts.items():
        print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser(description="Build metadata CSV for fiwGAN dataset")
    ap.add_argument('--data_root', required=True, help='Root directory containing language subfolders')
    ap.add_argument('--output', required=True, help='Output CSV path')
    ap.add_argument('--no_summary', action='store_true', help='Disable console summary')
    args = ap.parse_args()

    rows = build_metadata(args.data_root)
    write_csv(rows, args.output)
    if not args.no_summary:
        summarize(rows)


if __name__ == '__main__':
    main()
