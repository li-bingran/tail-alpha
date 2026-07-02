#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research Universe — 研究/验证用大股票池（S&P 500 成分快照）。

与 universe_manager.py（生产用 40-86 只池）分开：
  - 研究验证需要宽横截面（300-500 只）压低日度 CS-IC 噪声
  - 成分列表从 Wikipedia 拉取后落盘快照（data/sp500_snapshot.csv），保证可复现
  - 已知局限：用"当前"成分回测过去 2-3 年存在轻微幸存者偏差，报告中需注明

用法:
    from research_universe import load_sp500_universe, get_sp500_sector_map
    symbols = load_sp500_universe()          # 全部 ~500 只
    symbols = load_sp500_universe(top_n=300) # 按快照顺序取前 300
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
SNAPSHOT_PATH = DATA_DIR / 'sp500_snapshot.csv'

WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'


def _fetch_sp500_from_wikipedia() -> pd.DataFrame:
    """从 Wikipedia 拉取 S&P 500 成分表，返回 [symbol, sector] DataFrame。"""
    # urllib 在本机证书链下会 SSL 失败，用 requests（走 certifi）
    from io import StringIO
    import requests
    resp = requests.get(WIKI_URL, timeout=30,
                        headers={'User-Agent': 'Mozilla/5.0 (research script)'})
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]
    # yfinance 用 '-' 表示 share class（BRK.B → BRK-B）
    symbols = df['Symbol'].astype(str).str.strip().str.replace('.', '-', regex=False)
    out = pd.DataFrame({
        'symbol': symbols,
        'sector': df['GICS Sector'].astype(str).str.strip(),
    })
    out = out.dropna(subset=['symbol']).drop_duplicates(subset=['symbol'])
    out['snapshot_date'] = pd.Timestamp.today().strftime('%Y-%m-%d')
    return out


def load_sp500_snapshot(refresh: bool = False) -> pd.DataFrame:
    """加载（或首次拉取并落盘）S&P 500 成分快照。"""
    if SNAPSHOT_PATH.exists() and not refresh:
        return pd.read_csv(SNAPSHOT_PATH)

    df = _fetch_sp500_from_wikipedia()
    if len(df) < 400:
        raise RuntimeError(f'S&P 500 抓取异常，仅 {len(df)} 行，拒绝落盘')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SNAPSHOT_PATH, index=False, encoding='utf-8')
    print(f'[research_universe] 快照已保存: {SNAPSHOT_PATH} ({len(df)} 只)')
    return df


def load_sp500_universe(top_n: int | None = None, refresh: bool = False) -> list[str]:
    """返回 S&P 500 symbol 列表（按快照顺序，可截取前 top_n）。"""
    df = load_sp500_snapshot(refresh=refresh)
    symbols = df['symbol'].tolist()
    if top_n:
        symbols = symbols[:top_n]
    return symbols


def get_sp500_sector_map() -> dict[str, str]:
    """返回 {symbol: GICS sector} 映射（组合构建/行业中性用）。"""
    df = load_sp500_snapshot()
    return dict(zip(df['symbol'], df['sector']))


if __name__ == '__main__':
    import sys
    refresh = '--refresh' in sys.argv
    snap = load_sp500_snapshot(refresh=refresh)
    print(f'{len(snap)} 只, 快照日期: {snap["snapshot_date"].iloc[0]}')
    print(snap.groupby('sector').size().sort_values(ascending=False))
