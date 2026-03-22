#!/usr/bin/env python3
"""
Run a cutechess-cli H2H match and report results to the dashboard API.

Usage:
    python3 scripts/h2h-reporter.py \
        --engine1 /path/to/engine1 --tag1 v1.0.36 \
        --engine2 /path/to/engine2 --tag2 v1.0.37-rc1 \
        --games 5000 --tc "0/1:00+0.5" --concurrency 48 \
        --api-url https://chess-compete-production.up.railway.app \
        --api-key SECRET

The script:
1. Creates a match via the API
2. Runs a pilot phase (50 games with -debug) to measure NPS under full concurrency
3. Computes timemult = max(1.0, target_nps / measured_nps) and adjusts TC
4. Runs the main match, parsing output line by line
5. Reports results in batches every 10 games
6. Uploads the full PGN on completion
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import urllib.request
import urllib.error

TARGET_NPS = 2_000_000  # 2M NPS reference
PILOT_GAMES = 50


def api_request(url, api_key, endpoint, data=None):
    """Make an API request. Returns parsed JSON or None on error."""
    full_url = f"{url.rstrip('/')}{endpoint}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        full_url,
        data=body,
        headers={
            'Content-Type': 'application/json',
            'X-API-Key': api_key,
        },
        method='POST' if data else 'GET',
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"API error {e.code}: {e.read().decode()}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"API request failed: {e}", file=sys.stderr)
        return None


def build_cutechess_cmd(tag1, engine1, tag2, engine2, tc, concurrency, hash_mb,
                        threads, rounds, pgn_out, book='', book_format='pgn',
                        debug=False):
    """Build a cutechess-cli command list."""
    cmd = [
        'cutechess-cli',
        '-engine', f'name={tag1}', f'cmd={engine1}',
            f'option.Hash={hash_mb}', f'option.Threads={threads}',
        '-engine', f'name={tag2}', f'cmd={engine2}',
            f'option.Hash={hash_mb}', f'option.Threads={threads}',
        '-each', 'proto=uci', f'tc={tc}',
        '-repeat',
        '-games', '2',
        '-rounds', str(rounds),
        '-concurrency', str(concurrency),
        '-draw', 'movenumber=40', 'movecount=8', 'score=10',
        '-resign', 'movecount=3', 'score=500',
        '-pgnout', pgn_out,
        '-recover',
    ]
    if debug:
        cmd.append('-debug')
    if book:
        cmd.extend(['-openings', f'file={book}', f'format={book_format}',
                     'order=random', 'plies=24'])
    return cmd


def run_pilot(args):
    """Run a short pilot match with -debug to measure NPS under full concurrency.
    Returns average NPS across both engines, or None if measurement fails."""
    pilot_rounds = PILOT_GAMES // 2
    pilot_pgn = '/tmp/h2h_pilot.pgn'

    print(f"=== Pilot phase: {PILOT_GAMES} games at full concurrency with NPS measurement ===")
    cmd = build_cutechess_cmd(
        args.tag1, args.engine1, args.tag2, args.engine2,
        args.tc, args.concurrency, args.hash, args.threads,
        pilot_rounds, pilot_pgn, args.book, args.book_format,
        debug=True,
    )

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)

    # Parse NPS from debug output lines like:
    # <engine1(0)> info depth 12 ... nps 1823456 ...
    nps_re = re.compile(r'nps (\d+)')
    nps_samples = []

    for line in process.stdout:
        m = nps_re.search(line)
        if m:
            nps_samples.append(int(m.group(1)))

    process.wait()

    # Clean up pilot PGN
    if os.path.isfile(pilot_pgn):
        os.unlink(pilot_pgn)

    if not nps_samples:
        print("Warning: no NPS data captured during pilot", file=sys.stderr)
        return None

    avg_nps = sum(nps_samples) // len(nps_samples)
    print(f"Pilot complete: {len(nps_samples)} NPS samples, average {avg_nps:,} NPS")
    return avg_nps


def adjust_tc(tc_str, timemult):
    """Scale a time control string by timemult.
    Handles format like '0/1:00+0.5' -> '0/1:30+0.75' for timemult=1.5"""
    if timemult <= 1.0:
        return tc_str

    # Parse TC: "0/M:SS+I" or "0/S+I"
    m = re.match(r'^(\d+)/(?:(\d+):)?(\d+(?:\.\d+)?)\+(\d+(?:\.\d+)?)$', tc_str)
    if not m:
        print(f"Warning: cannot parse TC '{tc_str}' for adjustment", file=sys.stderr)
        return tc_str

    moves = m.group(1)
    minutes = int(m.group(2)) if m.group(2) else 0
    seconds = float(m.group(3))
    increment = float(m.group(4))

    total_seconds = minutes * 60 + seconds
    new_total = total_seconds * timemult
    new_increment = increment * timemult

    new_minutes = int(new_total // 60)
    new_seconds = new_total - new_minutes * 60

    if new_minutes > 0:
        if new_seconds == int(new_seconds):
            time_part = f"{new_minutes}:{int(new_seconds):02d}"
        else:
            time_part = f"{new_minutes}:{new_seconds:05.2f}"
    else:
        time_part = f"{new_seconds:.2f}".rstrip('0').rstrip('.')

    new_inc = f"{new_increment:.2f}".rstrip('0').rstrip('.')

    return f"{moves}/{time_part}+{new_inc}"


def upload_pgn_and_complete(api_url, api_key, match_id, pgn_out, label=""):
    """Upload PGN (if it exists) and ensure match is marked complete."""
    prefix = f"[{label}] " if label else ""
    pgn_uploaded = False
    if os.path.isfile(pgn_out):
        pgn_size = os.path.getsize(pgn_out)
        print(f"{prefix}Uploading PGN ({pgn_size / 1024 / 1024:.1f} MB)...")
        with open(pgn_out) as f:
            pgn_content = f.read()
        result = api_request(api_url, api_key, '/api/h2h/pgn', {
            'match_id': match_id,
            'pgn': pgn_content,
        })
        if result and result.get('ok'):
            print(f"{prefix}PGN uploaded and match marked complete.")
            pgn_uploaded = True
        else:
            print(f"{prefix}PGN upload failed.", file=sys.stderr)
    else:
        print(f"{prefix}No PGN file found at {pgn_out}", file=sys.stderr)

    if not pgn_uploaded:
        print(f"{prefix}Marking match as completed (without PGN).")
        api_request(api_url, api_key, '/api/h2h/pgn', {
            'match_id': match_id,
            'pgn': '',
        })


def run_match(args, match_id, tc, effective_tc):
    """Run the main match and report results."""
    rounds = args.games // 2

    cmd = build_cutechess_cmd(
        args.tag1, args.engine1, args.tag2, args.engine2,
        tc, args.concurrency, args.hash, args.threads,
        rounds, args.pgn_out, args.book, args.book_format,
    )

    print(f"Running main match: {args.games} games, TC {effective_tc}")
    print(f"Command: {' '.join(cmd)}")
    print()

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)

    # SIGTERM handler for spot instance reclamation (2-min warning)
    sigterm_received = False

    def handle_sigterm(signum, frame):
        nonlocal sigterm_received
        sigterm_received = True
        print("\n=== SIGTERM received (spot reclamation) - gracefully shutting down ===")
        process.terminate()

    signal.signal(signal.SIGTERM, handle_sigterm)

    batch_e1_wins = 0
    batch_e2_wins = 0
    batch_draws = 0
    batch_size = 0
    report_interval = 10
    total_reported = 0

    game_re = re.compile(
        r'Finished game \d+ \((\S+) vs (\S+)\): (1-0|0-1|1/2-1/2|\*)'
    )

    try:
        for line in process.stdout:
            line = line.rstrip()
            print(line)

            m = game_re.match(line)
            if m:
                white, black, result = m.group(1), m.group(2), m.group(3)
                if result == '1-0':
                    if white == args.tag1:
                        batch_e1_wins += 1
                    else:
                        batch_e2_wins += 1
                elif result == '0-1':
                    if black == args.tag1:
                        batch_e1_wins += 1
                    else:
                        batch_e2_wins += 1
                elif result == '1/2-1/2':
                    batch_draws += 1

                batch_size += 1

                if batch_size >= report_interval:
                    api_request(args.api_url, args.api_key, '/api/h2h/result', {
                        'match_id': match_id,
                        'engine1_wins': batch_e1_wins,
                        'engine2_wins': batch_e2_wins,
                        'draws': batch_draws,
                    })
                    total_reported += batch_size
                    batch_e1_wins = 0
                    batch_e2_wins = 0
                    batch_draws = 0
                    batch_size = 0

        process.wait()

        # Report remaining batch
        if batch_size > 0:
            api_request(args.api_url, args.api_key, '/api/h2h/result', {
                'match_id': match_id,
                'engine1_wins': batch_e1_wins,
                'engine2_wins': batch_e2_wins,
                'draws': batch_draws,
            })
            total_reported += batch_size

        if sigterm_received:
            print(f"\nSpot reclamation: {total_reported} games saved. Uploading partial PGN...")
            upload_pgn_and_complete(args.api_url, args.api_key, match_id, args.pgn_out,
                                   label="SIGTERM")
            sys.exit(0)

        print(f"\nMatch complete. {total_reported} games reported.")
        upload_pgn_and_complete(args.api_url, args.api_key, match_id, args.pgn_out)

    except KeyboardInterrupt:
        print("\nInterrupted. Reporting remaining results and uploading partial PGN...")
        process.kill()
        if batch_size > 0:
            api_request(args.api_url, args.api_key, '/api/h2h/result', {
                'match_id': match_id,
                'engine1_wins': batch_e1_wins,
                'engine2_wins': batch_e2_wins,
                'draws': batch_draws,
            })
        upload_pgn_and_complete(args.api_url, args.api_key, match_id, args.pgn_out,
                               label="interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        api_request(args.api_url, args.api_key, '/api/h2h/fail', {
            'match_id': match_id,
        })
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Run H2H match with API reporting')
    parser.add_argument('--engine1', required=True, help='Path to engine 1 binary')
    parser.add_argument('--tag1', required=True, help='Engine 1 version tag')
    parser.add_argument('--engine2', required=True, help='Path to engine 2 binary')
    parser.add_argument('--tag2', required=True, help='Engine 2 version tag')
    parser.add_argument('--games', type=int, default=5000, help='Total games')
    parser.add_argument('--tc', default='0/1:00+0.5', help='Time control')
    parser.add_argument('--concurrency', type=int, default=24, help='Concurrent games')
    parser.add_argument('--hash', type=int, default=128, help='Hash size MB')
    parser.add_argument('--threads', type=int, default=1, help='Threads per engine')
    parser.add_argument('--book', default='', help='Path to opening book')
    parser.add_argument('--book-format', default='pgn', help='Book format (pgn or epd)')
    parser.add_argument('--pgn-out', default='/tmp/h2h_match.pgn', help='PGN output file')
    parser.add_argument('--api-url', required=True, help='Dashboard API URL')
    parser.add_argument('--api-key', required=True, help='API key')
    parser.add_argument('--target-nps', type=int, default=TARGET_NPS,
                        help=f'Target NPS for timemult (default: {TARGET_NPS:,})')
    parser.add_argument('--no-pilot', action='store_true',
                        help='Skip pilot phase (no NPS calibration)')
    args = parser.parse_args()

    # Create match via API
    print(f"Creating match: {args.tag1} vs {args.tag2} ({args.games} games, TC {args.tc})")
    result = api_request(args.api_url, args.api_key, '/api/h2h/create', {
        'engine1_tag': args.tag1,
        'engine2_tag': args.tag2,
        'total_games': args.games,
        'time_control': args.tc,
    })
    if not result or 'match_id' not in result:
        print("Failed to create match", file=sys.stderr)
        sys.exit(1)

    match_id = result['match_id']
    print(f"Match created: ID {match_id}")

    # Pilot phase: measure NPS under full concurrency
    timemult = 1.0
    effective_tc = args.tc
    avg_nps = None

    if not args.no_pilot:
        avg_nps = run_pilot(args)
        if avg_nps:
            timemult = max(1.0, args.target_nps / avg_nps)
            if timemult > 1.0:
                effective_tc = adjust_tc(args.tc, timemult)
                print(f"NPS below target ({avg_nps:,} < {args.target_nps:,})")
                print(f"Applying timemult {timemult:.2f}: TC {args.tc} -> {effective_tc}")
            else:
                print(f"NPS OK ({avg_nps:,} >= {args.target_nps:,}), no TC adjustment needed")

            # Report calibration to API
            api_request(args.api_url, args.api_key, '/api/h2h/calibration', {
                'match_id': match_id,
                'avg_nps': avg_nps,
                'timemult': round(timemult, 2),
                'effective_tc': effective_tc,
            })
        print()

    # Run the main match
    run_match(args, match_id, effective_tc, effective_tc)


if __name__ == '__main__':
    main()
