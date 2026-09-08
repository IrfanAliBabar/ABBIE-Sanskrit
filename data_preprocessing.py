# data_preprocessing.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
@dataclass(frozen=True)
class PrepCfg:
    max_compound_len: int = 50
    window_len: int = 4
    min_len_diff: int = -2
    max_len_diff: int = 1
_SLP1_ALLOWED = set(
    list("aAiIuUeEoOfFxX") +
    ['k', 'K', 'g', 'G', 'N', 'c', 'C', 'j', 'J', 'Y', 'w', 'W', 'q', 'Q', 'R',
     't', 'T', 'd', 'D', 'n', 'p', 'P', 'b', 'B', 'm', 'y', 'r', 'l', 'v',
     'S', 'z', 's', 'h', 'L', '|'] +
    ['H', 'Z', 'V', 'M', '~', '/', '\\', '^', "'"]
)

def _sanitize_slp1(s: str) -> str:
    return "".join(ch for ch in s if ch in _SLP1_ALLOWED)

def _iast_to_slp1(text: str) -> str:
    #Direct transliteration: IAST -> SLP1.
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""
    slp = transliterate(t, sanscript.IAST, sanscript.SLP1)
    return _sanitize_slp1(slp)

def _lcp(a: str, b: str) -> int:
    
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def _lcs(a: str, b: str) -> int:
    
    n = min(len(a), len(b))
    i = 0
    while i < n and a[-1 - i] == b[-1 - i]:
        i += 1
    return i

def _choose_window(n: int, pref: int, suf: int, W: int) -> Optional[Tuple[int, int]]:
    """
    Build a window [lo, hi) of length W that covers the mismatch region:
      mismatch region in compound is [pref, n - suf)
    """
    mismatch_lo = pref
    mismatch_hi = n - suf
    if mismatch_hi < mismatch_lo:
        return None

    mismatch_len = mismatch_hi - mismatch_lo
    if mismatch_len > W:
        return None

    extra = W - mismatch_len
    left_pad = extra // 2
    right_pad = extra - left_pad

    lo = mismatch_lo - left_pad
    hi = mismatch_hi + right_pad

    lo = max(0, lo)
    hi = min(n, hi)

    cur = hi - lo
    if cur < W:
        need = W - cur
        take_left = min(lo, need)
        lo -= take_left
        need -= take_left
        take_right = min(n - hi, need)
        hi += take_right
        need -= take_right
        if need != 0:
            return None

    return lo, hi

def build_datalist_from_csv(csv_path: str, cfg: PrepCfg = PrepCfg()) -> List[list]:
    """
    Input CSV columns: compound, split  (IAST)
    Output rows:
      [w1_win, w2_win, comp_win, comp_full, lo, hi, w1_full, w2_full]
    """
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8").fillna("")

    # remove accidental repeated header rows
    if {"compound", "split"}.issubset(df.columns):
        df = df[~((df["compound"] == "compound") & (df["split"] == "split"))].copy()

    df = df[df["split"].astype(str).str.count(r"\+") == 1].copy()

    out: List[list] = []
    stats: Dict[str, int] = {"seen": 0, "kept": 0}

    for _, row in df.iterrows():
        stats["seen"] += 1

        comp_iast = str(row.get("compound", "")).strip()
        split_iast = str(row.get("split", "")).strip()

        parts = [p.strip() for p in split_iast.split("+")]
        if len(parts) != 2:
            continue
        p1_iast, p2_iast = parts
        if not p1_iast or not p2_iast or not comp_iast:
            continue

        w1 = _iast_to_slp1(p1_iast)
        w2 = _iast_to_slp1(p2_iast)
        comp = _iast_to_slp1(comp_iast)

        if not (w1 and w2 and comp):
            continue
        if not (cfg.window_len <= len(comp) <= cfg.max_compound_len):
            continue

        d = len(comp) - (len(w1) + len(w2))
        if not (cfg.min_len_diff <= d <= cfg.max_len_diff):
            continue

        pref = _lcp(comp, w1)
        suf = _lcs(comp, w2)

        win = _choose_window(len(comp), pref, suf, cfg.window_len)
        if win is None:
            continue
        lo, hi = win

        suffix_outside = len(comp) - hi
        if suffix_outside > len(w2):
            continue
        w2_prefix_len = len(w2) - suffix_outside

        # explicit consistency checks
        if comp[:lo] != w1[:lo]:
            continue
        if comp[hi:] != w2[w2_prefix_len:]:
            continue

        w1_win = w1[lo:]
        w2_win = w2[:w2_prefix_len]
        comp_win = comp[lo:hi]

        out.append([w1_win, w2_win, comp_win, comp, int(lo), int(hi), w1, w2])
        stats["kept"] += 1

    print(f"Rows seen: {stats['seen']}")
    print(f"Rows kept: {stats['kept']}")
    return out

def prepare_data(csv_path: str) -> List[list]:
    return build_datalist_from_csv(csv_path)

if __name__ == "__main__":
    pass
