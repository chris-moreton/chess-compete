# SPSA Worker Docker Image
#
# Builds a container that can run SPSA tuning games for rusty-rival.
# Communicates via HTTP API - no database credentials needed.
#
# Build:
#   docker build -t spsa-worker docker/
#
# Run:
#   docker run -e SPSA_API_URL=https://chess-compete-production.up.railway.app \
#              -e SPSA_API_KEY=your-api-key \
#              -e COMPUTER_NAME=worker-1 \
#              spsa-worker --concurrency 4

FROM rust:1.83-bookworm

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Clone repositories (both are public)
RUN git clone --depth 1 https://github.com/chris-moreton/rusty-rival.git
RUN git clone --depth 1 https://github.com/chris-moreton/chess-compete.git

# Set up Python virtual environment and install dependencies
WORKDIR /app/chess-compete
RUN python3 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -r requirements.txt

# Pre-compile rusty-rival to cache Rust dependencies (speeds up first iteration)
WORKDIR /app/rusty-rival
RUN cargo build --release

# Download and install Stockfish for reference games
# Using sse41-popcnt build for broad CPU compatibility (avx2/vnni builds fail on many systems)
WORKDIR /app
RUN curl -L https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-sse41-popcnt.tar -o stockfish.tar && \
    tar -xf stockfish.tar && \
    mv stockfish/stockfish-ubuntu-x86-64-sse41-popcnt /usr/local/bin/stockfish && \
    chmod +x /usr/local/bin/stockfish && \
    rm -rf stockfish.tar stockfish

# Update config.toml to use correct paths for Docker environment
WORKDIR /app/chess-compete
RUN sed -i 's|rusty_rival_path = .*|rusty_rival_path = "/app/rusty-rival"|' compete/spsa/config.toml && \
    sed -i 's|engine_path = .*|engine_path = "/usr/local/bin/stockfish"|' compete/spsa/config.toml

# Set environment variables
ENV PATH="/app/chess-compete/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Entry point
ENTRYPOINT ["python", "-m", "compete", "--spsa-http"]
CMD ["--concurrency", "1", "--spsa-batch", "10"]
