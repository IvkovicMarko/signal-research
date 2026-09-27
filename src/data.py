"""Loader for Ken French's industry portfolio data.

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Monthly value-weighted returns for US industry portfolios from 1926 onward. They are
constructed point-in-time, so unlike a list of today's index members they carry no
survivorship bias. Missing observations are coded as -99.99 and must be converted to
NaN, not left in.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
MISSING_CODES = (-99.99, -999.0)
_DATE_PATTERN = re.compile(r"^\d{6}$")


def _download(n_industries: int) -> str:
    url = f"{BASE_URL}/{n_industries}_Industry_Portfolios_CSV.zip"
    request = Request(url, headers={"User-Agent": "signal-research/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        return archive.read(name).decode("latin-1")


def _parse_first_monthly_block(text: str) -> pd.DataFrame:
    """Extract the leading value-weighted monthly block.

    The file holds several blocks separated by blank lines: value-weighted returns,
    then equal-weighted returns, then firm counts. Only the first is wanted. The
    parser is deliberately defensive because a silent mis-parse here would corrupt
    every downstream result while still producing plausible numbers.
    """
    lines = text.splitlines()

    first_data = next(
        (i for i, line in enumerate(lines) if _DATE_PATTERN.match(line.split(",")[0].strip())),
        None,
    )
    if first_data is None:
        raise ValueError("no monthly data rows found in the downloaded file")

    header_line = next(
        (lines[i] for i in range(first_data - 1, -1, -1) if lines[i].count(",") > 4),
        None,
    )
    if header_line is None:
        raise ValueError("could not locate the column header line")
    columns = [c.strip() for c in header_line.split(",")[1:]]

    index, rows = [], []
    for line in lines[first_data:]:
        if not line.strip():
            continue                      # tolerate stray blank lines inside the block
        fields = [f.strip() for f in line.split(",")]
        if not _DATE_PATTERN.match(fields[0]):
            break                         # a text heading means the next block started
        index.append(fields[0])
        rows.append([float(f) for f in fields[1:]])

    frame = pd.DataFrame(rows, columns=columns, index=pd.PeriodIndex(index, freq="M"))
    if frame.shape[1] != len(columns) or frame.shape[0] < 100:
        raise ValueError(f"unexpected parsed shape {frame.shape}; the parse likely failed")
    return frame


def load_industry_portfolios(n_industries: int = 49, use_cache: bool = True) -> pd.DataFrame:
    """Download (or load from cache) monthly value-weighted industry returns.

    Returns a DataFrame indexed by a monthly `PeriodIndex`, columns are industry
    names, values are returns as decimals.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{n_industries}_industry_portfolios.csv"

    if use_cache and cache_file.exists():
        text = cache_file.read_text(encoding="latin-1")
    else:
        text = _download(n_industries)
        if use_cache:
            # newline="" disables the platform newline translation that would
            # otherwise turn the file's CRLF endings into CRCRLF on Windows, which
            # reappears as a blank line between every data row on the way back in.
            cache_file.write_text(text, encoding="latin-1", newline="")

    frame = _parse_first_monthly_block(text)
    for code in MISSING_CODES:
        frame = frame.mask(np.isclose(frame, code))

    frame = frame / 100.0
    if not (frame.abs().max().max() < 5.0):
        raise ValueError("parsed returns look implausible; check the scaling")
    return frame


def to_price_index(returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative price index from a return panel, starting at 1.0."""
    return (1.0 + returns.fillna(0.0)).cumprod()
