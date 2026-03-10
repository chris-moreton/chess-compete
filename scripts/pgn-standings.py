#!/usr/bin/env python3
"""Show standings from a cutechess PGN file."""

import glob
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict


def score_to_elo(score):
    """Convert a win fraction (0-1) to Elo difference."""
    if score <= 0:
        return -999
    if score >= 1:
        return 999
    return -400 * math.log10(1 / score - 1)


def parse_pgn(content, results):
    games = re.findall(
        r'\[White "(.*?)"\].*?\[Black "(.*?)"\].*?\[Result "(.*?)"\]',
        content, re.DOTALL
    )
    count = 0
    for white, black, result in games:
        if result == '1-0':
            results[white]['w'] += 1; results[white]['pts'] += 1; results[white]['games'] += 1
            results[black]['l'] += 1; results[black]['games'] += 1
            count += 1
        elif result == '0-1':
            results[black]['w'] += 1; results[black]['pts'] += 1; results[black]['games'] += 1
            results[white]['l'] += 1; results[white]['games'] += 1
            count += 1
        elif result == '1/2-1/2':
            results[white]['d'] += 1; results[white]['pts'] += 0.5; results[white]['games'] += 1
            results[black]['d'] += 1; results[black]['pts'] += 0.5; results[black]['games'] += 1
            count += 1
    return count


def get_bayeselo_ratings(pgn_files):
    """Run bayeselo on PGN files and return {name: rating} dict."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bayeselo = os.path.join(script_dir, '..', 'bin', 'bayeselo')
    if not os.path.isfile(bayeselo):
        bayeselo = shutil.which('bayeselo')
    if not bayeselo:
        return None

    # Concatenate all PGNs into a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as tmp:
        for pgn_file in pgn_files:
            with open(pgn_file) as f:
                tmp.write(f.read())
                tmp.write('\n')
        tmp_path = tmp.name

    try:
        commands = f"readpgn {tmp_path}\nelo\nmm\nratings\nx\nx\n"
        result = subprocess.run(
            [bayeselo], input=commands, capture_output=True, text=True, timeout=30
        )
        ratings = {}
        for line in result.stdout.splitlines():
            # Parse lines like: "   1 v1.0.34-rc1   3152   0.0   35   35 ..."
            m = re.match(r'\s*\d+\s+(\S+)\s+(\d+)', line)
            if m:
                ratings[m.group(1)] = int(m.group(2))
        return ratings if ratings else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    finally:
        os.unlink(tmp_path)


def show_standings(pgn_files):
    results = defaultdict(lambda: {'w': 0, 'd': 0, 'l': 0, 'pts': 0.0, 'games': 0})
    total_games = 0

    for pgn_file in pgn_files:
        with open(pgn_file) as f:
            content = f.read()
        count = parse_pgn(content, results)
        total_games += count
        print(f'  {os.path.basename(pgn_file)}: {count} games')

    if not results:
        print('No games found.')
        return

    # Get BayesElo ratings
    bayeselo = get_bayeselo_ratings(pgn_files)

    # Calculate Elo for each engine relative to the weakest
    if bayeselo:
        sorted_engines = sorted(results, key=lambda e: bayeselo.get(e, 0), reverse=True)
    else:
        sorted_engines = sorted(results, key=lambda e: results[e]['pts'] / results[e]['games'], reverse=True)
    weakest = sorted_engines[-1]
    weakest_score = results[weakest]['pts'] / results[weakest]['games']
    base_elo = score_to_elo(weakest_score)

    has_bayes = bayeselo is not None
    if has_bayes:
        bayes_min = min(bayeselo.values())

    header = f'{"Engine":<20} {"Games":>6} {"W":>5} {"D":>5} {"L":>5} {"Pts":>7} {"Score":>7} {"Elo+":>6}'
    if has_bayes:
        header += f' {"BayesElo":>9} {"BayesElo+":>10}'
    width = len(header)

    print()
    print(f'Total games: {total_games} from {len(pgn_files)} file(s)')
    print()
    print(header)
    print('-' * width)
    for eng in sorted_engines:
        r = results[eng]
        score = r['pts'] / r['games'] if r['games'] > 0 else 0
        elo_diff = round(score_to_elo(score) - base_elo)
        line = f'{eng:<20} {r["games"]:>6} {r["w"]:>5} {r["d"]:>5} {r["l"]:>5} {r["pts"]:>7.1f} {score*100:>6.1f}% {elo_diff:>+5}'
        if has_bayes and eng in bayeselo:
            line += f' {bayeselo[eng]:>9} {bayeselo[eng] - bayes_min:>+9}'
        print(line)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Default: read all PGNs in results/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, '..', 'results')
        pgn_files = sorted(glob.glob(os.path.join(results_dir, '*.pgn')))
        if not pgn_files:
            print(f'No PGN files found in {results_dir}')
            sys.exit(1)
    else:
        pgn_files = sys.argv[1:]

    show_standings(pgn_files)
