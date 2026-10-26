"""Turtle Breakout 13: choose the out-of-sample universes by rule and download their data (free, data.binance.vision).

Stages (run in order; each is resumable and skips files that already exist):
    list      list every USD-M and spot USDT symbol in the archive (delisted included)
    screen    download daily klines, apply the TB 015 rules, draw the NEW sample, write the universe file
              notebooks/Turtle_Breakout/13/13_universe.json (commit it before running the notebook)
    download  1m klines (+ funding for NEW) for the chosen symbols

Output data: <BINANCE_UM_DIR or parent of DERIBIT_DATA_ROOT/Binance>/tb13/
    um/<SYM>_um_1m.parquet, funding/<SYM>_funding.parquet, spot/<SYM>_spot_1m.parquet, daily/..., lists/...

Usage (PC, Windows CMD):
    python tools\\download_tb13_data.py list
    python tools\\download_tb13_data.py screen
    python tools\\download_tb13_data.py download
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import turtle_oos as to  # noqa: E402

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DATA = "https://data.binance.vision"
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
         "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
KEEP = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"]
PREFIX = {"um": "data/futures/um/monthly/klines/", "spot": "data/spot/monthly/klines/"}
UNIVERSE_FILE = ROOT / "notebooks" / "Turtle_Breakout" / "13" / "13_universe.json"
THREADS = 12


def default_out() -> Path | None:
    v = os.getenv("BINANCE_UM_DIR")
    if v:
        return Path(v) / "tb13"
    root = os.getenv("DERIBIT_DATA_ROOT")
    if not root:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("DERIBIT_DATA_ROOT="):
                    root = line.split("=", 1)[1].strip().strip('"').strip("'")
    return Path(root).parent / "Binance" / "tb13" if root else None


def _safe(url: str) -> str:
    """Percent-encode non-ASCII characters in the URL path (some archive symbols are not ASCII)."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(parts._replace(path=urllib.parse.quote(parts.path, safe="/%")))


def fetch(url: str, tries: int = 3) -> bytes | None:
    for k in range(tries):
        try:
            with urllib.request.urlopen(_safe(url), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if k == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == tries - 1:
                raise
    return None


def s3_list(prefix: str, folders: bool) -> list[str]:
    """All CommonPrefixes (folders=True) or Keys under prefix, following the listing marker."""
    out, marker = [], ""
    while True:
        q = {"delimiter": "/", "prefix": prefix}
        if marker:
            q["marker"] = marker
        raw = fetch(f"{S3}?{urllib.parse.urlencode(q)}")
        if raw is None:
            break
        root = ET.fromstring(raw)
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        if folders:
            items = [e.findtext(f"{ns}Prefix") for e in root.findall(f"{ns}CommonPrefixes")]
        else:
            items = [e.findtext(f"{ns}Key") for e in root.findall(f"{ns}Contents")]
        out += [i for i in items if i]
        truncated = (root.findtext(f"{ns}IsTruncated") or "false").lower() == "true"
        if not truncated:
            break
        marker = root.findtext(f"{ns}NextMarker") or (out[-1] if out else "")
    return out


def verified(url: str) -> bytes | None:
    data = fetch(url)
    if data is None:
        return None
    chk = fetch(url + ".CHECKSUM")
    if chk is not None and hashlib.sha256(data).hexdigest() != chk.decode().split()[0].strip().lower():
        raise ValueError(f"checksum mismatch: {url}")
    return data


def parse_klines(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        b = z.read(z.namelist()[0])
    head = b.split(b"\n", 1)[0]
    df = pd.read_csv(io.BytesIO(b), header=0 if not head[:1].isdigit() else None)
    df.columns = KCOLS
    ts = df["open_time"].astype("int64")
    df["open_time"] = pd.to_datetime(ts, unit="us" if ts.max() > 10 ** 14 else "ms", utc=True)
    return df


def months_of(market: str, sym: str, interval: str) -> list[str]:
    keys = s3_list(f"{PREFIX[market]}{sym}/{interval}/", folders=False)
    return sorted(k for k in keys if k.endswith(".zip"))


# ---------------------------------------------------------------- stages

def stage_list(out: Path) -> None:
    (out / "lists").mkdir(parents=True, exist_ok=True)
    for market in ("um", "spot"):
        syms = sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in s3_list(PREFIX[market], folders=True))
        (out / "lists" / f"{market}_symbols.txt").write_text("\n".join(syms), encoding="utf-8")
        print(f"{market}: {len(syms)} symbols in the archive")


def daily_stats(market: str, sym: str, out: Path, op: to.OParams) -> dict | None:
    p = out / "daily" / market / f"{sym}_1d.parquet"
    if p.exists():
        d = pd.read_parquet(p)
    else:
        keys = months_of(market, sym, "1d")
        if market == "spot":
            keys = [k for k in keys if k[-11:-4] <= op.spot_end[:7]]
        parts = [parse_klines(r) for k in keys if (r := verified(f"{DATA}/{k}")) is not None]
        if not parts:
            return None
        d = pd.concat(parts, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
        d = d[["open_time", "close", "quote_volume"]].rename(columns={"open_time": "day"})
        p.parent.mkdir(parents=True, exist_ok=True); d.to_parquet(p, index=False)
    if market == "spot":
        d = d[(d.day >= pd.Timestamp(op.spot_start, tz="UTC")) & (d.day <= pd.Timestamp(op.spot_end, tz="UTC"))]
    d = to.cut_at_gap(d.reset_index(drop=True), op.max_gap_days)
    if not len(d):
        return None
    return {"symbol": sym, "first": str(d.day.iloc[0]), "last": str(d.day.iloc[-1]), "days": int(len(d)),
            "med_qvol": float(d.quote_volume.median())}


def stage_screen(out: Path, op: to.OParams) -> None:
    res = {}
    for market, kind, cand in (("um", "new", to.new_candidate), ("spot", "spot", to.spot_candidate)):
        syms = [s for s in (out / "lists" / f"{market}_symbols.txt").read_text(encoding="utf-8").split() if cand(s)]
        print(f"{market}: {len(syms)} candidates; downloading daily klines ...", flush=True)
        with ThreadPoolExecutor(THREADS) as ex:
            rows = [r for r in ex.map(lambda s: daily_stats(market, s, out, op), syms) if r]
        S = to.screen(pd.DataFrame(rows), op, kind)
        S.to_csv(out / "lists" / f"{kind}_screen.csv", index=False)
        elig = S.loc[S.eligible, "symbol"].tolist()
        chosen = to.draw(elig, op) if kind == "new" else sorted(elig)
        res[kind] = {"candidates": len(syms), "eligible": len(elig), "chosen": chosen,
                     "windows": {r.symbol: [r.first, r.last] for r in S[S.symbol.isin(chosen)].itertuples()}}
        print(f"{kind}: {len(elig)} eligible, {len(chosen)} chosen", flush=True)
    UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_FILE.write_text(json.dumps({"registration": to.registration(op), "created": str(date.today()), **res}, indent=1), encoding="utf-8")
    print("universe written:", UNIVERSE_FILE)


def build_1m(market: str, sym: str, out: Path, last_day: str) -> None:
    sub = "um" if market == "um" else "spot"
    p = out / sub / f"{sym}_{sub}_1m.parquet"
    if p.exists():
        return
    keys = [k for k in months_of(market, sym, "1m") if k[-11:-4] <= last_day[:7]]
    parts = []
    for k in keys:
        r = verified(f"{DATA}/{k}")
        if r is not None:
            parts.append(parse_klines(r)[KEEP])
    if not parts:
        print(f"{sym} {market}: no 1m data"); return
    df = pd.concat(parts, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df = df[df.open_time < pd.Timestamp(last_day, tz="UTC") + pd.Timedelta(days=1)]
    p.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(p, index=False)
    print(f"{sym} {market}: {len(df):,} rows {df.open_time.iloc[0]} -> {df.open_time.iloc[-1]}", flush=True)


def build_funding(sym: str, out: Path) -> None:
    p = out / "funding" / f"{sym}_funding.parquet"
    if p.exists():
        return
    keys = [k for k in s3_list(f"data/futures/um/monthly/fundingRate/{sym}/", folders=False) if k.endswith(".zip")]
    parts = []
    for k in keys:
        r = verified(f"{DATA}/{k}")
        if r is None:
            continue
        with zipfile.ZipFile(io.BytesIO(r)) as z:
            b = z.read(z.namelist()[0])
        has_header = not b.split(b"\n", 1)[0][:1].isdigit()
        f = pd.read_csv(io.BytesIO(b), header=0 if has_header else None)
        if has_header:
            f.columns = [str(c).strip().lower() for c in f.columns]
        else:
            f.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][:f.shape[1]]
        parts.append(f)
    if parts:
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_parquet(p, index=False)
    print(f"{sym} funding: {len(parts)} months", flush=True)


def stage_download(out: Path) -> None:
    U = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    jobs = [("um", s, w[1]) for s, w in U["new"]["windows"].items()] + [("spot", s, w[1]) for s, w in U["spot"]["windows"].items()]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda j: build_1m(j[0], j[1], out, j[2]), jobs))
        list(ex.map(lambda s: build_funding(s, out), U["new"]["chosen"]))
    print("done:", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["list", "screen", "download"])
    ap.add_argument("--out", type=Path, default=default_out())
    a = ap.parse_args()
    if a.out is None:
        ap.error("--out is required when DERIBIT_DATA_ROOT / BINANCE_UM_DIR are not set")
    a.out.mkdir(parents=True, exist_ok=True)
    op = to.OParams()
    {"list": lambda: stage_list(a.out), "screen": lambda: stage_screen(a.out, op), "download": lambda: stage_download(a.out)}[a.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
