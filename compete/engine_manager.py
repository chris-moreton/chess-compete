"""
Engine discovery, initialization, and management.
"""

import os
import platform
import stat
import subprocess
import sys
import traceback
import urllib.request
import urllib.error
from pathlib import Path

from compete.database import (
    db_retry,
    _get_app,
    load_elo_ratings,
    get_initial_elo,
    DB_ENABLED,
)
from compete.constants import DEFAULT_ELO


def get_engines_dir() -> Path:
    """Get the path to the engines directory."""
    # Allow override via environment variable
    if os.environ.get("ENGINES_DIR"):
        return Path(os.environ["ENGINES_DIR"])
    # Default to engines/ directory next to the compete package
    return Path(__file__).parent.parent / "engines"


def init_engine(engine_type: str, version: str) -> bool:
    """
    Download and initialize an engine from GitHub releases.

    Args:
        engine_type: "rusty" for rusty-rival or "java" for rivalchess-uci
        version: Version string (e.g., "v1.0.15" for rusty, "38" for java)

    Returns:
        True if successful, False otherwise
    """
    engine_dir = get_engines_dir()
    engine_dir.mkdir(parents=True, exist_ok=True)

    system = platform.system().lower()
    machine = platform.machine().lower()

    if engine_type == "rusty":
        # Normalize version to include 'v' prefix
        if not version.startswith("v"):
            version = f"v{version}"

        # Determine platform-specific asset name
        if system == "windows":
            asset_name = f"rusty-rival-{version}-windows-x86_64.exe"
        elif system == "darwin":
            if machine in ("arm64", "aarch64"):
                asset_name = f"rusty-rival-{version}-macos-aarch64"
            else:
                asset_name = f"rusty-rival-{version}-macos-x86_64"
        else:  # Linux
            asset_name = f"rusty-rival-{version}-linux-x86_64"

        download_url = f"https://github.com/chris-moreton/rusty-rival/releases/download/{version}/{asset_name}"
        target_dir = engine_dir / version
        target_file = target_dir / asset_name

    elif engine_type == "java":
        # Java version: "38" -> "v38.0.0"
        if version.startswith("v"):
            version = version[1:]  # Remove 'v' if present
        # Handle both "38" and "38.0.0" formats
        if "." not in version:
            full_version = f"v{version}.0.0"
        else:
            full_version = f"v{version}"

        asset_name = f"rivalchess-{full_version}.jar"
        download_url = f"https://github.com/chris-moreton/rivalchess-uci/releases/download/{full_version}/{asset_name}"
        target_dir = engine_dir / f"java-rival-{full_version[1:]}"  # e.g., java-rival-38.0.0
        target_file = target_dir / asset_name

    else:
        print(f"Error: Unknown engine type '{engine_type}'. Use 'rusty' or 'java'.")
        return False

    # Check if already exists
    if target_file.exists():
        print(f"Engine already exists: {target_file}")
        return True

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download the asset
    print(f"Downloading {asset_name} from {download_url}...")
    try:
        urllib.request.urlretrieve(download_url, target_file)
        print(f"Downloaded to {target_file}")
    except urllib.error.HTTPError as e:
        print(f"Error: Failed to download {download_url}")
        print(f"HTTP Error {e.code}: {e.reason}")
        # Clean up empty directory if download failed
        if target_dir.exists() and not any(target_dir.iterdir()):
            target_dir.rmdir()
        return False
    except Exception as e:
        print(f"Error: Failed to download: {e}")
        traceback.print_exc()
        if target_dir.exists() and not any(target_dir.iterdir()):
            target_dir.rmdir()
        return False

    # Post-download setup (not needed for .jar files)
    if not target_file.suffix == ".jar":
        if system == "darwin":
            # Remove macOS quarantine attribute
            print("Removing macOS quarantine attribute...")
            result = subprocess.run(
                ["xattr", "-d", "com.apple.quarantine", str(target_file)],
                capture_output=True
            )
            if result.returncode != 0:
                # xattr returns error if attribute doesn't exist, which is fine
                pass

        if system != "windows":
            # Make executable on Unix-like systems
            print("Setting executable permission...")
            target_file.chmod(target_file.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Engine initialized successfully: {target_dir.name}")
    return True


def init_stockfish() -> bool:
    """
    Download and initialize Stockfish from GitHub releases.

    Returns:
        True if successful, False otherwise
    """
    engine_dir = get_engines_dir()
    stockfish_dir = engine_dir / "stockfish"

    # Check if already exists
    if find_stockfish_binary(engine_dir):
        print("Stockfish already installed")
        return True

    stockfish_dir.mkdir(parents=True, exist_ok=True)

    system = platform.system().lower()
    machine = platform.machine().lower()

    # Stockfish 17 release assets
    base_url = "https://github.com/official-stockfish/Stockfish/releases/download/sf_17"

    if system == "windows":
        asset_name = "stockfish-windows-x86-64-avx2.zip"
        binary_name = "stockfish-windows-x86-64-avx2.exe"
    elif system == "darwin":
        if machine in ("arm64", "aarch64"):
            asset_name = "stockfish-macos-m1-apple-silicon.tar"
            binary_name = "stockfish-macos-m1-apple-silicon"
        else:
            asset_name = "stockfish-macos-x86-64-avx2.tar"
            binary_name = "stockfish-macos-x86-64-avx2"
    else:  # Linux
        asset_name = "stockfish-ubuntu-x86-64-avx2.tar"
        binary_name = "stockfish-ubuntu-x86-64-avx2"

    download_url = f"{base_url}/{asset_name}"
    archive_path = stockfish_dir / asset_name

    print(f"Downloading Stockfish from {download_url}...")
    try:
        urllib.request.urlretrieve(download_url, archive_path)
        print(f"Downloaded to {archive_path}")
    except Exception as e:
        print(f"Error: Failed to download Stockfish: {e}")
        traceback.print_exc()
        return False

    # Extract the archive
    print("Extracting...")
    try:
        import zipfile
        import tarfile

        if asset_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                # Find the binary in the archive
                for name in zf.namelist():
                    if name.endswith(".exe") and "stockfish" in name.lower():
                        zf.extract(name, stockfish_dir)
                        extracted = stockfish_dir / name
                        # Move to expected location if nested
                        if extracted.parent != stockfish_dir:
                            final_path = stockfish_dir / extracted.name
                            extracted.rename(final_path)
        else:  # .tar
            with tarfile.open(archive_path, 'r') as tf:
                for member in tf.getmembers():
                    if "stockfish" in member.name.lower() and member.isfile():
                        tf.extract(member, stockfish_dir)
                        extracted = stockfish_dir / member.name
                        # Move to expected location if nested
                        if extracted.parent != stockfish_dir:
                            final_path = stockfish_dir / extracted.name
                            extracted.rename(final_path)
                            # Set executable
                            final_path.chmod(final_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        # Clean up archive
        archive_path.unlink()

        # Verify installation
        if find_stockfish_binary(engine_dir):
            print("Stockfish installed successfully")
            return True
        else:
            print("Error: Stockfish binary not found after extraction")
            return False

    except Exception as e:
        print(f"Error extracting Stockfish: {e}")
        traceback.print_exc()
        return False


def parse_engine_name(name: str) -> tuple[str, str] | None:
    """
    Parse an engine name to determine its type and version.

    Returns:
        (engine_type, version) tuple, or None if unrecognized

    Examples:
        "v1.0.17" -> ("rusty", "v1.0.17")
        "java-rival-38" -> ("java", "38")
        "sf-2000" -> ("stockfish", "2000")
        "sf-full" -> ("stockfish", "full")
    """
    if len(name) >= 2 and name.startswith("v") and name[1:2].isdigit():
        return ("rusty", name)
    elif name.startswith("java-rival-"):
        version = name.replace("java-rival-", "")
        return ("java", version)
    elif name.startswith("sf-"):
        elo = name.replace("sf-", "")
        return ("stockfish", elo)
    return None


def ensure_engine_initialized(name: str, engine_dir: Path) -> bool:
    """
    Ensure a single engine is initialized locally.

    If not present, attempts to download and install it.

    Returns:
        True if engine is available, False otherwise
    """
    engines = discover_engines(engine_dir)

    if name in engines:
        return True

    print(f"Engine '{name}' not found locally, attempting to initialize...")

    parsed = parse_engine_name(name)
    if not parsed:
        print(f"Error: Cannot determine how to initialize '{name}'")
        print("  Expected formats: v1.0.17 (rusty-rival), java-rival-38, sf-2000")
        return False

    engine_type, version = parsed

    if engine_type == "stockfish":
        # For stockfish, we just need the binary installed
        if not init_stockfish():
            return False
        # Verify the virtual engine is now available
        engines = discover_engines(engine_dir)
        return name in engines
    elif engine_type == "rusty":
        return init_engine("rusty", version)
    elif engine_type == "java":
        return init_engine("java", version)

    return False


def ensure_engines_initialized(engine_names: list[str], engine_dir: Path) -> bool:
    """
    Ensure all specified engines are initialized locally.

    Returns:
        True if all engines are available, False if any failed
    """
    all_success = True
    for name in engine_names:
        if not ensure_engine_initialized(name, engine_dir):
            all_success = False
    return all_success


def find_stockfish_binary(engine_dir: Path) -> Path | None:
    """Find Stockfish binary, checking multiple possible names."""
    stockfish_dir = engine_dir / "stockfish"
    if not stockfish_dir.exists():
        return None

    # Try various Stockfish binary names
    candidates = [
        "stockfish-windows-x86-64-avx2.exe",  # Windows AVX2
        "stockfish-windows-x86-64.exe",       # Windows
        "stockfish.exe",                      # Windows generic
        "stockfish-macos-m1-apple-silicon",   # macOS ARM
        "stockfish-linux-x86_64",             # Linux x86_64
        "stockfish-ubuntu-x86-64-avx2",       # Linux AVX2
        "stockfish",                          # Generic name
    ]

    for name in candidates:
        binary = stockfish_dir / name
        if binary.exists():
            return binary

    return None


def discover_engines(engine_dir: Path) -> dict:
    """
    Auto-discover engines from directory structure.

    Convention:
    - v* directories: binary is {dir}/rusty-rival or versioned name
    - sf-XXXX: virtual Stockfish at Elo XXXX (uses stockfish binary with UCI_LimitStrength)
    - java-rival-*: looks for .jar file, runs with "java -jar"

    Returns dict: {engine_name: {"binary": Path, "uci_options": dict, "command": list (optional)}}

    If "command" key is present (for Java engines), use it for popen_uci.
    Otherwise, use "binary" path directly.
    """
    engines = {}
    stockfish_binary = find_stockfish_binary(engine_dir)

    if not engine_dir.exists():
        return engines

    for entry in engine_dir.iterdir():
        if not entry.is_dir():
            continue

        name = entry.name

        # Skip non-engine directories
        if name in ("stockfish", "syzygy", "epd", "maia-weights"):
            continue

        # v* engines (rusty-rival versions)
        if name.startswith("v") and len(name) > 1 and name[1:2].isdigit():
            binary = None

            # First try versioned binary names (e.g., rusty-rival-v1.0.13-windows-x86_64.exe)
            for f in entry.iterdir():
                if f.is_file() and f.name.startswith(f"rusty-rival-{name}"):
                    binary = f
                    break

            # Fall back to legacy names: rusty-rival.exe or rusty-rival
            if not binary:
                if (entry / "rusty-rival.exe").exists():
                    binary = entry / "rusty-rival.exe"
                elif (entry / "rusty-rival").exists():
                    binary = entry / "rusty-rival"

            if binary:
                engines[name] = {"binary": binary, "uci_options": {}}

        # java-rival engines (e.g., java-rival-36.0.0 -> java-rival-36)
        elif name.startswith("java-rival"):
            # Look for JAR files (cross-platform) or run.sh (legacy)
            jar_file = None
            for f in entry.iterdir():
                if f.is_file() and f.suffix == ".jar":
                    jar_file = f
                    break

            if jar_file:
                # Simplify name: java-rival-36.0.0 -> java-rival-36
                parts = name.split("-")
                if len(parts) >= 3:
                    version = parts[2].split(".")[0]  # "36.0.0" -> "36"
                    engine_name = f"java-rival-{version}"
                else:
                    engine_name = name
                # Store command as list for Java engines
                engines[engine_name] = {
                    "binary": jar_file,  # Keep for existence check
                    "command": ["java", "-jar", str(jar_file)],
                    "uci_options": {}
                }
            elif (entry / "run.sh").exists():
                # Legacy support for run.sh
                binary = entry / "run.sh"
                parts = name.split("-")
                if len(parts) >= 3:
                    version = parts[2].split(".")[0]
                    engine_name = f"java-rival-{version}"
                else:
                    engine_name = name
                engines[engine_name] = {"binary": binary, "uci_options": {}}

    # Add virtual Stockfish engines at different Elo levels
    if stockfish_binary:
        for elo in [1400, 1600, 1800, 2000, 2200, 2400, 2500, 2600, 2700, 2800, 3000]:
            name = f"sf-{elo}"
            engines[name] = {
                "binary": stockfish_binary,
                "uci_options": {
                    "UCI_LimitStrength": True,
                    "UCI_Elo": elo
                }
            }
        # Full-strength Stockfish
        engines["sf-full"] = {"binary": stockfish_binary, "uci_options": {}}

    return engines


def get_all_engines(engine_dir: Path) -> list[str]:
    """
    Get list of all available engines (auto-discovered).
    Returns sorted list of engine names.
    """
    engines = discover_engines(engine_dir)
    return sorted(engines.keys())


def _engine_matches_type(name: str, engine_type: str | None) -> bool:
    """Check if an engine name matches the specified type filter."""
    if engine_type is None:
        return True
    if engine_type == 'rusty':
        return name.startswith('v') and len(name) > 1 and name[1:2].isdigit()
    if engine_type == 'stockfish':
        return name.startswith('sf')
    return True


@db_retry
def get_active_engines(engine_dir: Path, engine_type: str = None,
                       include_inactive: bool = False) -> list[str]:
    """
    Get list of engines from database.

    Args:
        engine_dir: Path to engines directory
        engine_type: Filter by type ('rusty' or 'stockfish', None = all)
        include_inactive: If True, include inactive engines

    Returns:
        Sorted list of engine names matching the filters.
    """
    try:
        from web.database import db
        from web.models import Engine

        app = _get_app()
        with app.app_context():
            if include_inactive:
                engines = Engine.query.all()
            else:
                engines = Engine.query.filter_by(active=True).all()

            # Filter by engine type
            names = [e.name for e in engines if _engine_matches_type(e.name, engine_type)]
            return sorted(names)
    except Exception as e:
        print(f"Error: Failed to get engines from database: {e}")
        traceback.print_exc()
        sys.exit(1)


@db_retry
def set_engine_active(engine_name: str, active: bool, initial_elo: float | None = None) -> bool:
    """
    Enable or disable an engine by setting its active flag.
    If engine exists on disk but not in database, creates the database entry.
    Returns True if successful, False if engine not found on disk or in database.

    Args:
        engine_name: Name of the engine
        active: Whether to enable or disable the engine
        initial_elo: Optional starting Elo (if not provided, derived from engine name)
    """
    try:
        from web.database import db
        from web.models import Engine

        app = _get_app()
        with app.app_context():
            engine = Engine.query.filter_by(name=engine_name).first()
            if not engine:
                # Check if engine exists on disk
                engine_dir = get_engines_dir()
                discovered = discover_engines(engine_dir)
                if engine_name not in discovered:
                    return False
                # Create engine in database
                elo = initial_elo if initial_elo is not None else get_initial_elo(engine_name)
                engine = Engine(name=engine_name, active=active, initial_elo=int(elo))
                db.session.add(engine)
                db.session.commit()
                return True
            engine.active = active
            db.session.commit()
            return True
    except Exception as e:
        print(f"Error: Failed to update engine: {e}")
        traceback.print_exc()
        return False


@db_retry
def list_engines_status() -> list[tuple[str, bool, float, int]]:
    """
    List all engines with their active status, Elo, and games played.
    Returns list of (name, active, elo, games) tuples.
    """
    try:
        from web.database import db
        from web.models import Engine

        # Load calculated ELOs
        elo_ratings = load_elo_ratings()

        app = _get_app()
        with app.app_context():
            engines = Engine.query.all()
            result = []
            for e in engines:
                rating_data = elo_ratings.get(e.name, {"elo": float(e.initial_elo or 1500), "games": 0})
                result.append((e.name, e.active, rating_data["elo"], rating_data["games"]))
            return sorted(result, key=lambda x: -x[2])  # Sort by Elo descending
    except Exception as e:
        print(f"Error: Failed to list engines: {e}")
        traceback.print_exc()
        return []


def get_engine_info(name: str, engine_dir: Path) -> tuple[Path | list, dict]:
    """
    Get engine command and UCI options.
    Returns (command, uci_options_dict).

    command can be:
    - Path: for native executables (will be converted to str for popen_uci)
    - list: for Java engines ["java", "-jar", "path/to/engine.jar"]

    If the engine is not found locally, attempts to download and initialize it.
    """
    engines = discover_engines(engine_dir)

    if name not in engines:
        # Engine not found locally - attempt to download it
        print(f"Engine '{name}' not found locally, attempting to download...")
        if ensure_engine_initialized(name, engine_dir):
            # Re-discover engines after successful initialization
            engines = discover_engines(engine_dir)

        if name not in engines:
            print(f"Error: Engine '{name}' not found and could not be initialized")
            print(f"Available engines: {', '.join(get_all_engines(engine_dir))}")
            sys.exit(1)

    engine_config = engines[name]
    binary_path = engine_config["binary"]
    uci_options = engine_config.get("uci_options", {})

    if not binary_path.exists():
        print(f"Error: Engine binary not found: {binary_path}")
        sys.exit(1)

    # Return command list if present (for Java engines), otherwise return path
    command = engine_config.get("command", binary_path)
    return command, uci_options


def resolve_engine_name(shorthand: str, engine_dir: Path) -> str:
    """
    Resolve a shorthand engine name to the full engine name.
    E.g., 'v1' -> 'v001-baseline', 'v10' -> 'v010-arrayvec-movelist'

    If engine is not found locally, attempts to download/initialize it first.
    """
    engines = discover_engines(engine_dir)

    # If exact match exists, use it
    if shorthand in engines:
        return shorthand

    matches = []

    # Handle v-prefixed numeric shorthand (v1 -> v001, v10 -> v010)
    if shorthand.startswith("v") and len(shorthand) > 1 and shorthand[1:].isdigit():
        num = int(shorthand[1:])
        # Try different zero-padded formats
        for fmt in [f"v{num:03d}-", f"v{num:02d}-", f"v{num}-"]:
            matches += [name for name in engines.keys() if name.startswith(fmt)]

    # Look for matches starting with the shorthand followed by '-'
    if not matches:
        matches = [name for name in engines.keys() if name.startswith(f"{shorthand}-")]

    # Only look for dot-suffix matches if shorthand contains a dot
    if '.' in shorthand:
        matches += [name for name in engines.keys() if name.startswith(f"{shorthand}.")]

    # Deduplicate
    matches = list(set(matches))

    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        # Engine not found locally - try to initialize it
        if ensure_engine_initialized(shorthand, engine_dir):
            # Re-discover engines after initialization
            engines = discover_engines(engine_dir)
            if shorthand in engines:
                return shorthand
            # Check for matches again (in case initialization created a differently-named engine)
            matches = [name for name in engines.keys() if name.startswith(f"{shorthand}")]
            if len(matches) == 1:
                return matches[0]

        print(f"Error: No engine found matching '{shorthand}'")
        print(f"Available engines: {', '.join(sorted(engines.keys()))}")
        sys.exit(1)
    else:
        print(f"Error: Ambiguous engine name '{shorthand}' matches: {', '.join(sorted(matches))}")
        sys.exit(1)
