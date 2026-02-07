# SPSA Worker Docker Image

Run SPSA tuning workers in Docker containers on any machine.

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
cd docker/

# Configure
cp .env.example .env
# Edit .env with your API key

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Scale to multiple workers
docker-compose up -d --scale worker=3

# Stop
docker-compose down
```

### Option 2: Docker Run

```bash
# Build the image
docker build -t spsa-worker docker/

# Run a worker
docker run -d \
  -e SPSA_API_URL=https://chess-compete-production.up.railway.app \
  -e SPSA_API_KEY=your-api-key \
  -e COMPUTER_NAME=worker-1 \
  --name spsa-worker-1 \
  spsa-worker --concurrency 4

# Check logs
docker logs -f spsa-worker-1

# Stop
docker stop spsa-worker-1
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `SPSA_API_URL` | Chess-compete API endpoint | `https://chess-compete-production.up.railway.app` |
| `SPSA_API_KEY` | API authentication key | (required) |
| `COMPUTER_NAME` | Worker identifier for logs | `docker-worker` |
| `CONCURRENCY` | Parallel games per container | `4` |

## Command Line Options

The container runs `python -m compete --spsa-http` with these options:

- `--concurrency N` / `-c N`: Number of parallel games (default: 1)
- `--poll-interval N` / `-p N`: Seconds between API polls when idle (default: 10)

Example:
```bash
docker run spsa-worker --concurrency 8
```

## Resource Requirements

Each worker container needs:
- **CPU**: ~1 core per concurrent game (so `-c 4` needs ~4 cores)
- **Memory**: ~1GB base + ~256MB per concurrent game
- **Disk**: ~2GB for Rust toolchain, repos, and compiled engines

## Building on Different Architectures

The default Dockerfile uses x86-64 Stockfish. For ARM64 (Apple Silicon, AWS Graviton):

```dockerfile
# Replace the Stockfish download line with:
RUN curl -L https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-linux-arm64.tar -o stockfish.tar
```

## Troubleshooting

### "unauthorized" error
- Check your `SPSA_API_KEY` is correct
- Verify the API key is set on the server (`SPSA_WORKER_API_KEY` env var)

### No work available (dots printing)
- This is normal when no SPSA iterations are pending
- The master needs to create iterations for workers to process

### Build failures
- Ensure you have enough disk space (~3GB)
- Check network connectivity to GitHub and crates.io

### Engine compilation errors
- The Rust toolchain is included in the image
- Compilation happens inside the container, not on the host
