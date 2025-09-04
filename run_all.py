# run_all.py
import argparse
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from scraper import fetch_and_bucket_by_day
from analyze import Config, TrendEngine, save_outputs
from visualize import plot_heatmap

def parse_args():
    p = argparse.ArgumentParser(description="Pulsegen – Topic Trend Agent")
    p.add_argument("--app", required=True, help="Play Store link or app id (e.g., in.swiggy.android)")
    p.add_argument("--target", default=datetime.now().date().isoformat(),
                   help="Target date (YYYY-MM-DD). Report covers T-30..T.")
    p.add_argument("--backfill_from", default=None,
                   help="(Optional) Backfill start date YYYY-MM-DD (e.g., 2024-06-01). If set, will fetch and bucket reviews up to --target.")
    p.add_argument("--max_fetch", type=int, default=4000, help="Max newest reviews to fetch for bucketing.")
    p.add_argument("--no_fetch", action="store_true", help="Skip fetching; use existing data/ day files.")
    return p.parse_args()

def main():
    args = parse_args()
    T = datetime.fromisoformat(args.target).date()
    window_start = (T - timedelta(days=30)).isoformat()

    print(f"📆 Target date: {T} | Window: {window_start} .. {T}")

    # 1) Fetch & bucket (optional)
    if not args.no_fetch:
        if args.backfill_from:
            print(f"🔄 Backfilling {args.backfill_from} .. {args.target}")
            fetch_and_bucket_by_day(args.app, args.backfill_from, args.target, max_fetch=args.max_fetch)
        else:
            # minimal pull that still buckets into last 31 days
            print(f"🔄 Fetching newest reviews and bucketing days in {window_start} .. {args.target}")
            fetch_and_bucket_by_day(args.app, window_start, args.target, max_fetch=args.max_fetch)
    else:
        print("⏭️ Skipping fetch (using existing data/*.csv).")

    # 2) Analyze (agentic pipeline)
    cfg = Config()
    engine = TrendEngine(cfg)
    counts, meta = engine.run(target_date=args.target, horizon_days=30)

    if counts.empty:
        print("⚠️ No data to report. Check data/ files or increase --max_fetch.")
        return

    # 3) Save outputs
    save_outputs(counts, meta, target_date=args.target)

    # 4) Visualize
    csv_path = os.path.join(cfg.output_dir, f"report_{args.target}.csv")
    plot_heatmap(csv_path)

    print("✅ Done.")

if __name__ == "__main__":
    main()
