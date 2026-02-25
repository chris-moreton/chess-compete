#!/usr/bin/env bash
set -euo pipefail

# Chess-Compete Mac Native SPSA Worker Setup
# Usage: curl -sS https://raw.githubusercontent.com/chris-moreton/chess-compete/main/scripts/setup-mac-worker.sh | bash

COMPETE_REPO="https://github.com/chris-moreton/chess-compete.git"
RIVAL_REPO="https://github.com/chris-moreton/rusty-rival.git"
COMPETE_DIR="$HOME/chess-compete"
RIVAL_DIR="$HOME/rusty-rival"
API_URL="https://chess-compete-production.up.railway.app"

# --- Name generation ---

ADJECTIVES=(
  "azure" "bold" "calm" "deft" "eager" "fair" "grand" "hale" "keen" "lucid"
  "merry" "noble" "prime" "quick" "rapid" "sharp" "stoic" "swift" "vivid" "wise"
  "amber" "brave" "clear" "crisp" "dusty" "fleet" "frost" "gleam" "hardy" "ivory"
  "jolly" "lunar" "maple" "ocean" "pearl" "quiet" "regal" "silky" "tidal" "urban"
)

NOUNS=(
  "charm" "blade" "creek" "drift" "ember" "flame" "grove" "haven" "jewel" "knoll"
  "larch" "marsh" "nexus" "orbit" "prism" "quill" "ridge" "spark" "tower" "vault"
  "acorn" "birch" "cedar" "delta" "fjord" "glyph" "heron" "inlet" "lance" "mirth"
  "oasis" "plume" "reign" "shore" "trail" "vigor" "aspen" "brook" "coral" "forge"
)

generate_name() {
  local adj_idx=$((RANDOM % ${#ADJECTIVES[@]}))
  local noun_idx=$((RANDOM % ${#NOUNS[@]}))
  echo "${ADJECTIVES[$adj_idx]}-${NOUNS[$noun_idx]}"
}

# --- Detect architecture ---

ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
  echo ""
  echo "=========================================="
  echo "  Chess-Compete Mac Worker Setup (ARM64)"
  echo "=========================================="
elif [ "$ARCH" = "x86_64" ]; then
  echo ""
  echo "=========================================="
  echo "  Chess-Compete Mac Worker Setup (Intel)"
  echo "=========================================="
else
  echo "Error: Unsupported architecture: $ARCH"
  exit 1
fi
echo ""

# --- Check for existing directories ---

for dir in "$COMPETE_DIR" "$RIVAL_DIR"; do
  if [ -d "$dir" ]; then
    echo "Directory '$dir' already exists."
    printf "Remove it and start fresh? [y/N] "
    read -r confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
      rm -rf "$dir"
    else
      echo "Aborting."
      exit 1
    fi
  fi
done

# --- Install prerequisites via Homebrew ---

if ! command -v brew &>/dev/null; then
  echo "Homebrew not found. Installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add brew to PATH for Apple Silicon
  if [ "$ARCH" = "arm64" ] && [ -f /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
fi

echo "Checking prerequisites..."

if ! command -v rustc &>/dev/null; then
  echo "Installing Rust..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi

if ! command -v python3 &>/dev/null; then
  echo "Installing Python 3..."
  brew install python3
fi

if ! command -v git &>/dev/null; then
  echo "Installing git..."
  brew install git
fi

echo "  Rust:    $(rustc --version)"
echo "  Python:  $(python3 --version)"
echo "  Git:     $(git --version)"
echo ""

# --- Clone repos ---

echo "Cloning rusty-rival..."
git clone --depth 1 "$RIVAL_REPO" "$RIVAL_DIR"
echo ""

echo "Cloning chess-compete..."
git clone --depth 1 "$COMPETE_REPO" "$COMPETE_DIR"
echo ""

# --- Install Stockfish ---

STOCKFISH_DIR="$RIVAL_DIR/engines"
mkdir -p "$STOCKFISH_DIR"

echo "Installing Stockfish..."
if [ "$ARCH" = "x86_64" ]; then
  # Intel Mac — download pre-built binary
  SF_URL="https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-macos-x86-64-modern.tar"
  TMPDIR_SF=$(mktemp -d)
  echo "  Downloading pre-built Stockfish for Intel..."
  curl -sL "$SF_URL" -o "$TMPDIR_SF/stockfish.tar"
  tar -xf "$TMPDIR_SF/stockfish.tar" -C "$TMPDIR_SF"
  # Find the binary inside the extracted archive
  SF_BIN=$(find "$TMPDIR_SF" -name "stockfish" -type f -perm +111 | head -1)
  if [ -z "$SF_BIN" ]; then
    echo "Error: Could not find Stockfish binary in download."
    exit 1
  fi
  cp "$SF_BIN" "$STOCKFISH_DIR/stockfish"
  chmod +x "$STOCKFISH_DIR/stockfish"
  rm -rf "$TMPDIR_SF"
else
  # Apple Silicon — build from source
  echo "  Building Stockfish from source for Apple Silicon (this may take a few minutes)..."
  TMPDIR_SF=$(mktemp -d)
  git clone --depth 1 https://github.com/official-stockfish/Stockfish.git "$TMPDIR_SF/Stockfish"
  cd "$TMPDIR_SF/Stockfish/src"
  make -j"$(sysctl -n hw.ncpu)" build ARCH=apple-silicon
  cp stockfish "$STOCKFISH_DIR/stockfish"
  cd "$HOME"
  rm -rf "$TMPDIR_SF"
fi

echo "  Stockfish installed at $STOCKFISH_DIR/stockfish"
echo ""

# --- Set up Python venv ---

echo "Setting up Python virtual environment..."
cd "$COMPETE_DIR"
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
echo "  Python dependencies installed."
echo ""

# --- Patch config.toml ---

echo "Configuring SPSA settings..."
CONFIG_FILE="$COMPETE_DIR/compete/spsa/config.toml"

# Set rusty_rival_path to absolute path
sed -i '' "s|^rusty_rival_path = .*|rusty_rival_path = \"$RIVAL_DIR\"|" "$CONFIG_FILE"

# Set reference engine path to absolute path
sed -i '' "s|^engine_path = .*|engine_path = \"$STOCKFISH_DIR/stockfish\"|" "$CONFIG_FILE"

# Disable S3 build cache (no AWS credentials needed)
sed -i '' 's|^s3_build_cache = .*|s3_build_cache = ""|' "$CONFIG_FILE"

echo "  Config updated: $CONFIG_FILE"
echo ""

# --- Collect settings ---

printf "Enter your API key: "
read -r api_key

if [ -z "$api_key" ]; then
  echo "Error: API key is required."
  exit 1
fi

default_name="mac-$(generate_name)"
printf "Enter a name for this worker [%s]: " "$default_name"
read -r worker_name
worker_name="${worker_name:-$default_name}"

# --- Create .env ---

cat > "$COMPETE_DIR/.env" <<EOF
SPSA_API_URL=$API_URL
SPSA_API_KEY=$api_key
DATABASE_URL=
EOF

echo ""
echo "  .env created at $COMPETE_DIR/.env"
echo ""

# --- Pre-compile rusty-rival ---

echo "Building rusty-rival (optimized for this CPU)..."
echo "  This may take a few minutes on first build..."
cd "$RIVAL_DIR"
RUSTFLAGS="-C target-cpu=native" cargo build --release
echo "  Build complete."
echo ""

# --- Detect CPU count and suggest concurrency ---

CPU_COUNT=$(sysctl -n hw.ncpu)
# Suggest leaving 1-2 cores free
if [ "$CPU_COUNT" -le 4 ]; then
  SUGGESTED_CORES=$((CPU_COUNT - 1))
else
  SUGGESTED_CORES=$((CPU_COUNT - 2))
fi
[ "$SUGGESTED_CORES" -lt 1 ] && SUGGESTED_CORES=1

echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "  Worker name:  $worker_name"
echo "  CPU cores:    $CPU_COUNT detected ($SUGGESTED_CORES suggested for workers)"
echo ""
echo "  To start the worker:"
echo ""
echo "    cd $COMPETE_DIR && COMPUTER_NAME=$worker_name .venv/bin/python -m compete --spsa-http --auto-timemult -c $SUGGESTED_CORES"
echo ""
echo "  To run in the background:"
echo ""
echo "    cd $COMPETE_DIR && nohup env COMPUTER_NAME=$worker_name .venv/bin/python -m compete --spsa-http --auto-timemult -c $SUGGESTED_CORES > worker.log 2>&1 &"
echo ""
echo "  To stop a background worker:"
echo ""
echo "    kill \$(pgrep -f 'compete --spsa-http')"
echo ""
echo "  To update later:"
echo ""
echo "    cd $COMPETE_DIR && git pull"
echo "    cd $RIVAL_DIR && git pull && RUSTFLAGS=\"-C target-cpu=native\" cargo build --release"
echo ""
