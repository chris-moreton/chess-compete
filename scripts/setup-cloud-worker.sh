#!/bin/bash
#
# SPSA Cloud Worker Setup Script
#
# This script sets up a fresh Ubuntu/Debian VM to run SPSA workers.
# Run as root or with sudo on a new cloud instance.
#
# Usage:
#   curl -sSL <raw-script-url> | sudo bash
#   # OR
#   sudo bash setup-cloud-worker.sh
#
# After setup, configure .env and start the worker:
#   cd /opt/chess-compete
#   nano .env  # Add DATABASE_URL
#   sudo systemctl start spsa-worker
#

set -e

# Configuration
INSTALL_DIR="/opt"
CHESS_COMPETE_REPO="https://github.com/chris-moreton/chess-compete.git"
RUSTY_RIVAL_REPO="https://github.com/chris-moreton/rusty-rival.git"
WORKER_USER="spsa"
CONCURRENCY=4  # Adjust based on VM cores

echo "========================================"
echo "SPSA Cloud Worker Setup"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo bash $0)"
    exit 1
fi

# Detect number of CPUs
NUM_CPUS=$(nproc)
echo "Detected $NUM_CPUS CPU cores"
CONCURRENCY=$((NUM_CPUS > 1 ? NUM_CPUS : 1))
echo "Will use concurrency: $CONCURRENCY"

echo ""
echo "Step 1: Installing system dependencies..."
echo "----------------------------------------"
apt-get update
apt-get install -y \
    build-essential \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    pkg-config \
    libssl-dev

echo ""
echo "Step 2: Creating worker user..."
echo "----------------------------------------"
if id "$WORKER_USER" &>/dev/null; then
    echo "User $WORKER_USER already exists"
else
    useradd -m -s /bin/bash "$WORKER_USER"
    echo "Created user $WORKER_USER"
fi

echo ""
echo "Step 3: Installing Rust..."
echo "----------------------------------------"
if [ -d "/home/$WORKER_USER/.cargo" ]; then
    echo "Rust already installed"
else
    su - "$WORKER_USER" -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
fi

echo ""
echo "Step 4: Cloning repositories..."
echo "----------------------------------------"
cd "$INSTALL_DIR"

if [ -d "chess-compete" ]; then
    echo "chess-compete already exists, pulling latest..."
    cd chess-compete && git pull && cd ..
else
    git clone "$CHESS_COMPETE_REPO"
fi

if [ -d "rusty-rival" ]; then
    echo "rusty-rival already exists, pulling latest..."
    cd rusty-rival && git pull && cd ..
else
    git clone "$RUSTY_RIVAL_REPO"
fi

# Set ownership
chown -R "$WORKER_USER:$WORKER_USER" chess-compete rusty-rival

echo ""
echo "Step 5: Setting up Python environment..."
echo "----------------------------------------"
cd "$INSTALL_DIR/chess-compete"
su - "$WORKER_USER" -c "cd $INSTALL_DIR/chess-compete && python3 -m venv .venv"
su - "$WORKER_USER" -c "cd $INSTALL_DIR/chess-compete && .venv/bin/pip install --upgrade pip"
su - "$WORKER_USER" -c "cd $INSTALL_DIR/chess-compete && .venv/bin/pip install -r requirements.txt"

echo ""
echo "Step 6: Creating .env template..."
echo "----------------------------------------"
if [ ! -f "$INSTALL_DIR/chess-compete/.env" ]; then
    cat > "$INSTALL_DIR/chess-compete/.env" << 'EOF'
# Database connection string
# Replace with your actual PostgreSQL connection string
DATABASE_URL=postgresql://username:password@hostname:5432/database_name
EOF
    chown "$WORKER_USER:$WORKER_USER" "$INSTALL_DIR/chess-compete/.env"
    chmod 600 "$INSTALL_DIR/chess-compete/.env"
    echo "Created .env template - YOU MUST EDIT THIS!"
else
    echo ".env already exists"
fi

echo ""
echo "Step 7: Creating systemd service..."
echo "----------------------------------------"
cat > /etc/systemd/system/spsa-worker.service << EOF
[Unit]
Description=SPSA Chess Engine Tuning Worker
After=network.target

[Service]
Type=simple
User=$WORKER_USER
WorkingDirectory=$INSTALL_DIR/chess-compete
Environment="PATH=$INSTALL_DIR/chess-compete/.venv/bin:/home/$WORKER_USER/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
Environment="RUST_MIN_STACK=4097152"
ExecStart=$INSTALL_DIR/chess-compete/.venv/bin/python -m compete --spsa -c $CONCURRENCY
Restart=always
RestartSec=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=spsa-worker

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
echo "Created systemd service: spsa-worker"

echo ""
echo "Step 8: Creating helper scripts..."
echo "----------------------------------------"

# Status script
cat > "$INSTALL_DIR/chess-compete/worker-status.sh" << 'EOF'
#!/bin/bash
echo "=== SPSA Worker Status ==="
systemctl status spsa-worker --no-pager
echo ""
echo "=== Recent Logs ==="
journalctl -u spsa-worker -n 20 --no-pager
EOF
chmod +x "$INSTALL_DIR/chess-compete/worker-status.sh"

# Logs script
cat > "$INSTALL_DIR/chess-compete/worker-logs.sh" << 'EOF'
#!/bin/bash
journalctl -u spsa-worker -f
EOF
chmod +x "$INSTALL_DIR/chess-compete/worker-logs.sh"

# Update script
cat > "$INSTALL_DIR/chess-compete/worker-update.sh" << 'EOF'
#!/bin/bash
set -e
echo "Stopping worker..."
sudo systemctl stop spsa-worker

echo "Updating repositories..."
cd /opt/chess-compete && git pull
cd /opt/rusty-rival && git pull

echo "Updating Python dependencies..."
cd /opt/chess-compete
.venv/bin/pip install -r requirements.txt

echo "Starting worker..."
sudo systemctl start spsa-worker

echo "Done! Check status with: ./worker-status.sh"
EOF
chmod +x "$INSTALL_DIR/chess-compete/worker-update.sh"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Configure the database connection:"
echo "   sudo nano $INSTALL_DIR/chess-compete/.env"
echo "   # Set DATABASE_URL to your PostgreSQL connection string"
echo ""
echo "2. Start the worker:"
echo "   sudo systemctl start spsa-worker"
echo "   sudo systemctl enable spsa-worker  # Start on boot"
echo ""
echo "3. Monitor the worker:"
echo "   cd $INSTALL_DIR/chess-compete"
echo "   ./worker-status.sh   # Quick status check"
echo "   ./worker-logs.sh     # Follow live logs"
echo ""
echo "4. Update the worker:"
echo "   cd $INSTALL_DIR/chess-compete"
echo "   ./worker-update.sh"
echo ""
echo "Worker will run with $CONCURRENCY parallel games."
echo "To change, edit: /etc/systemd/system/spsa-worker.service"
echo ""
