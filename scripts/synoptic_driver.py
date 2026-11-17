#!/usr/bin/env python3
"""
Driver for synoptic_bulk_download.py. Works out which year-months are still missing and
runs the downloader for them.

  python synoptic_driver.py                 download the next missing month
  python synoptic_driver.py --year 2019     download every missing month in 2019 (per-year runner)
  python synoptic_driver.py --dry-run       show what would run, download nothing
  python synoptic_driver.py --retry-failed  clear the "given up" marks so those months are tried again

A month counts as done when every state's obs_hourly file for it exists. The downloader only
writes a file after all of that state-month's batches succeed, so an existing file is complete.

A month that fails gets two attempts, counted across runs. After the second failure it is marked
"failed" and skipped in later runs. Failures are reported at the end of each run. Attempt
counts and failure reasons are kept in <out>/logs/driver_state.json.

Run it with the analysis_env Python so the downloader can use pyarrow:
  /home/admoore/.venv/analysis_env/bin/python synoptic_driver.py
"""

import argparse
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DOWNLOADER = os.path.join(HERE, "synoptic_bulk_download.py")

# Range of months to cover. The end is the last complete month before the current one.
DEFAULT_START = "2015-01"
DEFAULT_END = "2026-10"

# A month is given up after this many failed attempts (counted across runs).
MAX_ATTEMPTS = 2
# Stop the run after this many failures in a row. Several months failing back to back usually
# means an outage or a bad token, not bad data, so there is no point burning attempts on every month.
MAX_CONSECUTIVE_FAILURES = 3


def load_downloader():
    spec = importlib.util.spec_from_file_location("synoptic_bulk_download", DOWNLOADER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_month(s):
    year, month = s.split("-")
    return int(year), int(month)


def months_between(start, end):
    """Yield (year, month) tuples from start to end, inclusive."""
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def month_key(year, month):
    return f"{year}-{month:02d}"


def month_bounds(year, month):
    first = dt.date(year, month, 1)
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    return first, nxt - dt.timedelta(days=1)


def month_done(mod, out_dir, year, month, states):
    win_start = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
    return all(os.path.exists(mod.obs_path(out_dir, win_start, s)) for s in states)


def load_state(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def run_month(year, month, log_file):
    """Run the downloader for one month, echoing its output to the console and the log.

    Returns (exit_code, last_output_line).
    """
    first, last = month_bounds(year, month)
    cmd = [sys.executable, DOWNLOADER, "--start", first.isoformat(), "--end", last.isoformat()]
    print(f"\n=== {month_key(year, month)}: {' '.join(cmd[1:])}", flush=True)
    log_file.write(f"\n=== {month_key(year, month)} started {dt.datetime.now().isoformat(timespec='seconds')}\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last_line = ""
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        if line.strip():
            last_line = line.strip()
    log_file.flush()
    return proc.wait(), last_line


def acquire_lock(lock_path):
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f"Another driver run appears to be active (lock: {lock_path}). "
                 "If no run is active, delete the lock file and retry.")
    os.write(fd, f"{os.getpid()} {time.ctime()}\n".encode())
    os.close(fd)


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, help="Download every missing month in this year (per-year runner)")
    p.add_argument("--start", default=DEFAULT_START, help="First month to cover, YYYY-MM (default %(default)s)")
    p.add_argument("--end", default=DEFAULT_END, help="Last month to cover, YYYY-MM (default %(default)s)")
    p.add_argument("--dry-run", action="store_true", help="List the months that would run and exit")
    p.add_argument("--retry-failed", action="store_true",
                   help="Clear the given-up marks so those months get two new attempts")
    args = p.parse_args(argv)

    mod = load_downloader()
    out_dir = mod.DEFAULT_OUT_DIR
    states = mod.STATES
    logs_dir = os.path.join(out_dir, "logs")
    state_path = os.path.join(logs_dir, "driver_state.json")

    months = list(months_between(parse_month(args.start), parse_month(args.end)))
    if args.year is not None:
        months = [(y, m) for y, m in months if y == args.year]
        if not months:
            sys.exit(f"Year {args.year} is outside the configured range {args.start} to {args.end}")

    if args.dry_run:
        state = load_state(state_path)
    else:
        os.makedirs(logs_dir, exist_ok=True)
        state = load_state(state_path)

    if args.retry_failed and not args.dry_run:
        cleared = [k for k, v in state.items() if v.get("status") == "failed"]
        for k in cleared:
            state[k]["attempts"] = 0
            state[k]["status"] = "retry"
        save_state(state_path, state)
        print(f"Cleared given-up status for {len(cleared)} month(s): {', '.join(sorted(cleared)) or 'none'}")

    missing = [(y, m) for y, m in months if not month_done(mod, out_dir, y, m, states)]
    eligible = [(y, m) for y, m in missing if state.get(month_key(y, m), {}).get("status") != "failed"]
    given_up = [(y, m) for y, m in missing if (y, m) not in eligible]

    print(f"Output: {out_dir}")
    print(f"Range {month_key(*months[0])} to {month_key(*months[-1])}: "
          f"{len(months) - len(missing)} of {len(months)} months complete, {len(missing)} missing, "
          f"{len(given_up)} given up")

    if not eligible:
        print("Nothing to download.")
        report_failures(state)
        return 0

    if args.dry_run:
        # Default mode would try the first eligible month, moving on if it fails.
        for y, m in (eligible if args.year is not None else eligible[:1]):
            print(f"would download {month_key(y, m)}")
        report_failures(state)
        return 0

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"driver_{stamp}.log")
    lock_path = os.path.join(logs_dir, "driver.lock")
    acquire_lock(lock_path)
    any_failed = False
    consecutive_failures = 0
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            for y, m in eligible:
                key = month_key(y, m)
                rc, last_line = run_month(y, m, log_file)
                ok = rc == 0 and month_done(mod, out_dir, y, m, states)
                entry = state.get(key, {"attempts": 0})
                if ok:
                    consecutive_failures = 0
                    state.pop(key, None)
                    save_state(state_path, state)
                    print(f"{key} complete.")
                    log_file.write(f"{key} complete.\n")
                    if args.year is None:
                        break  # default mode: stop after one successful month
                    continue

                any_failed = True
                consecutive_failures += 1
                reason = last_line or f"downloader exit code {rc}"
                if rc == 0:
                    reason = "finished but not all state files exist"
                entry["attempts"] = entry.get("attempts", 0) + 1
                entry["last_error"] = reason
                entry["last_attempt"] = dt.datetime.now().isoformat(timespec="seconds")
                entry["status"] = "failed" if entry["attempts"] >= MAX_ATTEMPTS else "retry"
                state[key] = entry
                save_state(state_path, state)
                msg = (f"{key} failed (attempt {entry['attempts']} of {MAX_ATTEMPTS}): {reason}"
                       + ("; given up, skipping" if entry["status"] == "failed" else "; will retry later"))
                print("\n" + msg)
                log_file.write(msg + "\n")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    stop = (f"{consecutive_failures} months failed in a row; stopping this run. "
                            "Check the token and API status before rerunning. Attempt counts were kept.")
                    print("\n" + stop)
                    log_file.write(stop + "\n")
                    break
                # Otherwise keep going: the next month is tried in this run.

        remaining = [(y, m) for y, m in months if not month_done(mod, out_dir, y, m, states)]
        print(f"\nDone. Log: {log_path}. Months still missing in range: {len(remaining)}.")
        report_failures(state)
        return 1 if any_failed else 0
    finally:
        os.remove(lock_path)


def report_failures(state):
    """Print months given up on and months that failed once, with their last error."""
    given_up = [(k, v) for k, v in sorted(state.items()) if v.get("status") == "failed"]
    retrying = [(k, v) for k, v in sorted(state.items()) if v.get("status") == "retry"]
    if given_up:
        print(f"\nGiven up after {MAX_ATTEMPTS} failed attempts ({len(given_up)}); "
              "run with --retry-failed to try again:")
        for k, v in given_up:
            print(f"  {k}: {v.get('last_error', '?')}  (last attempt {v.get('last_attempt', '?')})")
    if retrying:
        print(f"\nFailed once, will retry on the next run ({len(retrying)}):")
        for k, v in retrying:
            print(f"  {k}: {v.get('last_error', '?')}")
    if not given_up and not retrying:
        print("\nNo failures recorded.")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
