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
2. Runs cutechess-cli, parsing output line by line
3. Reports results in batches every 10 games
4. Uploads the full PGN on completion
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error


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
    args = parser.parse_args()

    rounds = args.games // 2  # cutechess plays 2 games per round (colour swap)

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

    # Build cutechess command
    cmd = [
        'cutechess-cli',
        '-engine', f'name={args.tag1}', f'cmd={args.engine1}',
            f'option.Hash={args.hash}', f'option.Threads={args.threads}',
        '-engine', f'name={args.tag2}', f'cmd={args.engine2}',
            f'option.Hash={args.hash}', f'option.Threads={args.threads}',
        '-each', 'proto=uci', f'tc={args.tc}',
        '-repeat',
        '-games', '2',
        '-rounds', str(rounds),
        '-concurrency', str(args.concurrency),
        '-draw', 'movenumber=40', 'movecount=8', 'score=10',
        '-resign', 'movecount=3', 'score=500',
        '-pgnout', args.pgn_out,
        '-recover',
    ]

    if args.book:
        cmd.extend(['-openings', f'file={args.book}', f'format={args.book_format}',
                     'order=random', 'plies=24'])

    print(f"Running: {' '.join(cmd)}")
    print()

    # Run cutechess-cli and parse output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)

    # Track results for batch reporting
    batch_e1_wins = 0
    batch_e2_wins = 0
    batch_draws = 0
    batch_size = 0
    report_interval = 10
    total_reported = 0

    # Parse cutechess output lines like:
    # Finished game 1 (v1.0.36 vs v1.0.37-rc1): 1-0 {White wins by adjudication}
    # Score of v1.0.36 vs v1.0.37-rc1: 5 - 3 - 2  [0.600] 10
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

        print(f"\nMatch complete. {total_reported} games reported.")

        # Upload PGN
        if os.path.isfile(args.pgn_out):
            pgn_size = os.path.getsize(args.pgn_out)
            print(f"Uploading PGN ({pgn_size / 1024 / 1024:.1f} MB)...")
            with open(args.pgn_out) as f:
                pgn_content = f.read()
            result = api_request(args.api_url, args.api_key, '/api/h2h/pgn', {
                'match_id': match_id,
                'pgn': pgn_content,
            })
            if result and result.get('ok'):
                print("PGN uploaded successfully.")
            else:
                print("PGN upload failed.", file=sys.stderr)
        else:
            print(f"Warning: PGN file not found at {args.pgn_out}", file=sys.stderr)
            api_request(args.api_url, args.api_key, '/api/h2h/fail', {
                'match_id': match_id,
            })

    except KeyboardInterrupt:
        print("\nInterrupted. Marking match as failed.")
        process.kill()
        api_request(args.api_url, args.api_key, '/api/h2h/fail', {
            'match_id': match_id,
        })
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        api_request(args.api_url, args.api_key, '/api/h2h/fail', {
            'match_id': match_id,
        })
        sys.exit(1)


if __name__ == '__main__':
    main()
