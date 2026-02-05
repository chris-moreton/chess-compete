#!/usr/bin/env bash
set -euo pipefail

# Chess-Compete SPSA Worker Installer
# Usage: curl -sS https://raw.githubusercontent.com/chris-moreton/chess-compete/main/scripts/install-worker.sh | bash

REPO_URL="https://github.com/chris-moreton/chess-compete.git"
INSTALL_DIR="chess-compete"

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

# --- Preflight checks ---

echo ""
echo "==================================="
echo "  Chess-Compete SPSA Worker Setup"
echo "==================================="
echo ""

for cmd in git docker docker-compose; do
  if ! command -v "$cmd" &>/dev/null; then
    # docker-compose might be a docker subcommand
    if [ "$cmd" = "docker-compose" ] && docker compose version &>/dev/null; then
      continue
    fi
    echo "Error: '$cmd' is not installed. Please install it and try again."
    exit 1
  fi
done

if [ -d "$INSTALL_DIR" ]; then
  echo "Directory '$INSTALL_DIR' already exists."
  printf "Remove it and start fresh? [y/N] "
  read -r confirm
  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    rm -rf "$INSTALL_DIR"
  else
    echo "Aborting."
    exit 1
  fi
fi

# --- Clone ---

echo "Cloning chess-compete..."
git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
echo ""

# --- Collect settings ---

printf "Enter your API key: "
read -r api_key

if [ -z "$api_key" ]; then
  echo "Error: API key is required."
  exit 1
fi

default_name=$(generate_name)
printf "Enter a name for this worker [%s]: " "$default_name"
read -r worker_name
worker_name="${worker_name:-$default_name}"

# --- Configure .env ---

cp "$INSTALL_DIR/docker/.env.example" "$INSTALL_DIR/docker/.env"

if [[ "$OSTYPE" == "darwin"* ]]; then
  sed -i '' "s|^SPSA_API_KEY=.*|SPSA_API_KEY=${api_key}|" "$INSTALL_DIR/docker/.env"
  sed -i '' "s|^COMPUTER_NAME=.*|COMPUTER_NAME=${worker_name}|" "$INSTALL_DIR/docker/.env"
else
  sed -i "s|^SPSA_API_KEY=.*|SPSA_API_KEY=${api_key}|" "$INSTALL_DIR/docker/.env"
  sed -i "s|^COMPUTER_NAME=.*|COMPUTER_NAME=${worker_name}|" "$INSTALL_DIR/docker/.env"
fi

echo ""
echo "Configuration saved to $INSTALL_DIR/docker/.env"

# --- Build ---

echo ""
echo "Building Docker image (this may take a few minutes)..."
echo ""

cd "$INSTALL_DIR/docker"

if docker compose version &>/dev/null; then
  docker compose build
else
  docker-compose build
fi

echo ""
echo "==================================="
echo "  Setup complete!"
echo "==================================="
echo ""
echo "  Worker name: $worker_name"
echo ""
echo "  To start the worker:"
echo "    cd $INSTALL_DIR/docker && docker-compose up -d"
echo ""
echo "  To follow the logs:"
echo "    cd $INSTALL_DIR/docker && docker-compose logs -f"
echo ""
echo "  To scale to multiple workers:"
echo "    cd $INSTALL_DIR/docker && docker-compose up -d --scale worker=3"
echo ""
echo "  To stop:"
echo "    cd $INSTALL_DIR/docker && docker-compose down"
echo ""
