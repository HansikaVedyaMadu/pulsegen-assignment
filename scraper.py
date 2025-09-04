# scraper.py
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

from google_play_scraper import reviews, Sort

IST = timezone(timedelta(hours=5, minutes=30))

def _parse_app_id_from_link(link_or_id: str) -> str:
    """Accepts full Play Store URL or bare app id and returns app id."""
    if "http" in link_or_id and "details?id=" in link_or_id:
        import urllib.parse as up
        parsed = up.urlparse(link_or_id)
        qs = up.parse_qs(parsed.query)
        return qs.get("id", [""])[0]
    return link_or_id

def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _utc_to_date_ist(dt_utc: datetime) -> str:
    return dt_utc.astimezone(IST).date().isoformat()

def fetch_and_bucket_by_day(app_id_or_link: str, start_date: str, end_date: str, data_dir="data",
                            max_fetch: int = 4000) -> Dict[str, str]:
    """
    Fetch up to `max_fetch` newest reviews and bucket them per-day (IST).
    Saves CSVs at data/YYYY-MM-DD.csv for any day within [start_date, end_date].
    Returns {date: filepath} for days written.
    """
    app_id = _parse_app_id_from_link(app_id_or_link)
    _ensure_dir(data_dir)

    # Pull a big batch (sorted by NEWEST). We filter by date locally.
    print(f"🔎 Fetching up to {max_fetch} newest reviews for {app_id} …")
    rvws, _ = reviews(
        app_id,
        lang="en",
        country="in",
        sort=Sort.NEWEST,
        count=max_fetch,
    )

    if not rvws:
        print("⚠️ No reviews returned.")
        return {}

    # Normalize and bucket
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    buckets: Dict[str, List[Dict[str, Any]]] = {}

    for r in rvws:
        at: datetime = r.get("at")  # timezone-aware UTC
        if not isinstance(at, datetime):
            continue
        day = at.date()
        if start <= day <= end:
            key = _utc_to_date_ist(at)
            buckets.setdefault(key, []).append({
                "reviewId": r.get("reviewId"),
                "userName": r.get("userName"),
                "score": r.get("score"),
                "content": r.get("content") or "",
                "at_utc": at.isoformat(),
            })

    written = {}
    for day, rows in sorted(buckets.items()):
        df = pd.DataFrame(rows)
        fp = os.path.join(data_dir, f"{day}.csv")
        df.to_csv(fp, index=False)
        written[day] = fp

    if written:
        print(f"✅ Saved {len(written)} daily file(s) into {data_dir}/ (range {min(written)} … {max(written)})")
    else:
        print("⚠️ No reviews fell within the requested date range.")
    return written
