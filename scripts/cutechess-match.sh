#!/bin/bash
set -euo pipefail

ENGINES_DIR="$(cd "$(dirname "$0")/../engines" && pwd)"
OPENINGS_DIR="$(cd "$(dirname "$0")/../openings" && pwd)"
RESULTS_DIR="$(cd "$(dirname "$0")/../results" && pwd)"

ENGINES=()
ROUNDS=100
TC="0/1:00+0.5"
HASH=128
BOOK="8moves_v3.pgn"
BOOK_FORMAT="pgn"
CONCURRENCY=1
THREADS=1

usage() {
    echo "Usage: $0 <engine1> <engine2> [engine3 ...] [options]"
    echo ""
    echo "Arguments:"
    echo "  engine1, engine2, ...  Engine version directories under engines/ (2 or more)"
    echo ""
    echo "Options:"
    echo "  -r, --rounds N      Number of rounds (default: $ROUNDS)"
    echo "  -t, --tc TC         Time control (default: $TC)"
    echo "  -h, --hash N        Hash in MB (default: $HASH)"
    echo "  -b, --book FILE     Opening book filename in openings/ (default: $BOOK)"
    echo "  -f, --format FMT    Book format: pgn or epd (default: $BOOK_FORMAT)"
    echo "  -c, --concurrency N Concurrent games (default: $CONCURRENCY)"
    echo "  --threads N         UCI Threads per engine (default: $THREADS)"
    echo "  --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 v1.0.32-rc3 v1.0.31 -r 50 -t '0/2:00+1'"
    echo "  $0 v1.0.34-rc1 v1.0.33 v1.0.32 v1.0.30 --threads 1 -c 8 -r 200"
    exit 1
}

# Parse args: collect engines until we hit an option flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--rounds)     ROUNDS="$2"; shift 2 ;;
        -t|--tc)         TC="$2"; shift 2 ;;
        -h|--hash)       HASH="$2"; shift 2 ;;
        -b|--book)       BOOK="$2"; shift 2 ;;
        -f|--format)     BOOK_FORMAT="$2"; shift 2 ;;
        -c|--concurrency) CONCURRENCY="$2"; shift 2 ;;
        --threads)       THREADS="$2"; shift 2 ;;
        --help)          usage ;;
        -*)              echo "Unknown option: $1"; usage ;;
        *)               ENGINES+=("$1"); shift ;;
    esac
done

[[ ${#ENGINES[@]} -lt 2 ]] && { echo "Error: need at least 2 engines"; usage; }

# Find engine binaries
find_engine() {
    local dir="$ENGINES_DIR/$1"
    if [[ ! -d "$dir" ]]; then
        echo "Error: engine directory not found: $dir" >&2
        exit 1
    fi
    local bin
    bin=$(find "$dir" -maxdepth 1 -type f -perm +111 | head -1)
    if [[ -z "$bin" ]]; then
        echo "Error: no executable found in $dir" >&2
        exit 1
    fi
    echo "$bin"
}

# Build engine flags
ENGINE_FLAGS=()
for eng in "${ENGINES[@]}"; do
    cmd=$(find_engine "$eng")
    ENGINE_FLAGS+=(-engine "name=$eng" "cmd=$cmd" "option.Hash=$HASH" "option.Threads=$THREADS")
done

# Verify book exists
BOOK_PATH="$OPENINGS_DIR/$BOOK"
if [[ ! -f "$BOOK_PATH" ]]; then
    echo "Error: opening book not found: $BOOK_PATH" >&2
    exit 1
fi

# Tournament type: round-robin for 3+ engines, default (match) for 2
TOURNAMENT_FLAGS=()
if [[ ${#ENGINES[@]} -gt 2 ]]; then
    TOURNAMENT_FLAGS+=(-tournament round-robin)
fi

# Output file
mkdir -p "$RESULTS_DIR"
ENGINES_LABEL=$(IFS=_; echo "${ENGINES[*]}")
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
PGN_OUT="$RESULTS_DIR/${ENGINES_LABEL}_${TIMESTAMP}.pgn"

echo "Engines: ${ENGINES[*]}"
echo "Tournament: $([ ${#ENGINES[@]} -gt 2 ] && echo "round-robin" || echo "match")"
echo "Time control: $TC"
echo "Rounds: $ROUNDS"
echo "Book: $BOOK ($BOOK_FORMAT)"
echo "Output: $PGN_OUT"
echo ""

cutechess-cli \
    "${ENGINE_FLAGS[@]}" \
    "${TOURNAMENT_FLAGS[@]}" \
    -each proto=uci tc="$TC" \
    -openings file="$BOOK_PATH" format="$BOOK_FORMAT" order=random plies=24 \
    -repeat \
    -games 2 \
    -rounds "$ROUNDS" \
    -concurrency "$CONCURRENCY" \
    -draw movenumber=40 movecount=8 score=10 \
    -resign movecount=3 score=500 \
    -pgnout "$PGN_OUT" \
    -recover
