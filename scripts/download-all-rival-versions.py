#!/usr/bin/env python3
"""
Fetch the list of all published Rusty Rival releases from GitHub, compare
against what's already in engines/, and download whatever is missing.

Usage:
    python scripts/download-all-rival-versions.py
    python scripts/download-all-rival-versions.py --dry-run
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
GITHUB_REPO = "chris-moreton/rusty-rival"


def cpu_supports_avx2() -> bool:
    """Mirrors compete.engine_manager.cpu_supports_avx2(). Conservative: if we
    cannot tell, assume no - an AVX2 binary on a non-AVX2 CPU dies with SIGILL."""
    system = platform.system().lower()
    try:
        if system == "linux":
            with open("/proc/cpuinfo") as fh:
                flags = fh.read().replace("\n", " ")
            return " avx2 " in f" {flags} "
        if system == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.leaf7_features"],
                capture_output=True, text=True, timeout=5,
            ).stdout.lower()
            return "avx2" in out
    except Exception:
        pass
    return False


def asset_candidates(version: str) -> list[str]:
    """Mirrors compete.engine_manager.rusty_asset_names(): preferred asset first.

    The -avx2 build is preferred where the CPU supports it - the plain x86_64
    asset is built for baseline x86-64 (no POPCNT/SSE4/AVX) and is markedly
    slower, which for a testing harness distorts which changes look like wins.
    Set RIVAL_ASSET_VARIANT=baseline|avx2 to pin one."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return [f"rusty-rival-{version}-macos-aarch64"]
        return [f"rusty-rival-{version}-macos-x86_64"]
    if system == "windows":
        baseline = f"rusty-rival-{version}-windows-x86_64.exe"
        avx2 = f"rusty-rival-{version}-windows-x86_64-avx2.exe"
    else:
        baseline = f"rusty-rival-{version}-linux-x86_64"
        avx2 = f"rusty-rival-{version}-linux-x86_64-avx2"
    variant = os.environ.get("RIVAL_ASSET_VARIANT", "auto").lower()
    if variant == "baseline":
        return [baseline]
    if variant == "avx2":
        return [avx2]
    return [avx2, baseline] if cpu_supports_avx2() else [baseline]


def expected_asset_name(version: str, available: set = None) -> str:
    """The asset this machine wants for `version`.

    When the release's asset list is known, return the best candidate it
    actually publishes - releases predating the -avx2 artifacts only offer the
    baseline name. Older releases (roughly v0.0.1-v1.0.12) used different
    naming/archive schemes that init_engine() has never supported - see
    is_supported_release() below."""
    candidates = asset_candidates(version)
    if available:
        for candidate in candidates:
            if candidate in available:
                return candidate
    return candidates[-1]


def get_engines_dir() -> Path:
    """Mirrors compete.engine_manager.get_engines_dir() without importing the
    compete package - see run-latest-rival-match.py for why."""
    if os.environ.get("ENGINES_DIR"):
        return Path(os.environ["ENGINES_DIR"])
    return REPO_ROOT / "engines"


def find_python_interpreter() -> str:
    """Prefer the repo's .venv interpreter so `--init` has requirements.txt installed."""
    if ".venv" in Path(sys.executable).parts:
        return sys.executable
    for candidate in (REPO_ROOT / ".venv" / "bin" / "python", REPO_ROOT / ".venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def fetch_all_releases(repo: str) -> list[dict]:
    """Fetch all published (non-draft) releases from GitHub, oldest first.
    Each entry is {'tag': str, 'assets': set[str]} - the asset list lets us tell
    upfront whether a release uses a naming scheme init_engine() understands,
    without a separate API call per tag."""
    releases = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "chess-compete-downloader",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            print(f"Error: GitHub API request failed: {e}")
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"Error: could not reach GitHub API: {e}")
            sys.exit(1)

        if not data:
            break
        releases.extend(
            {"tag": r["tag_name"], "assets": {a["name"] for a in r.get("assets", [])}}
            for r in data if not r.get("draft")
        )
        if len(data) < 100:
            break
        page += 1

    return list(reversed(releases))  # API returns newest-first; we want oldest-first


def find_existing_rusty_rival_binaries(engine_dir: Path) -> dict:
    """Map version -> set of rusty-rival binary filenames already present.

    Returning the filenames rather than a bare "is it downloaded" flag lets the
    caller decide per release WHICH asset it wants: a directory holding only the
    baseline binary is incomplete on an AVX2 machine, but complete for a release
    that never published an -avx2 artifact."""
    if not engine_dir.exists():
        return {}
    existing = {}
    for d in engine_dir.iterdir():
        if not d.is_dir() or not (d.name.startswith("v") and len(d.name) > 1 and d.name[1].isdigit()):
            continue
        existing[d.name] = {
            f.name for f in d.iterdir() if f.is_file() and f.name.startswith(f"rusty-rival-{d.name}")
        }
    return existing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what's missing, but don't download anything")
    args = parser.parse_args()

    engine_dir = get_engines_dir()

    print(f"Fetching release list from github.com/{GITHUB_REPO}...")
    releases = fetch_all_releases(GITHUB_REPO)
    print(f"Found {len(releases)} published release(s) on GitHub")

    existing = find_existing_rusty_rival_binaries(engine_dir)

    have, missing, unsupported = [], [], []
    for r in releases:
        tag = r["tag"]
        want = expected_asset_name(tag, r["assets"])
        if want not in r["assets"]:
            unsupported.append(tag)
        elif want in existing.get(tag, set()):
            have.append(tag)
        else:
            missing.append(tag)

    print(f"Already downloaded: {len(have)}/{len(releases)}")

    if unsupported:
        print(f"\nSkipping {len(unsupported)} release(s) that use an asset naming scheme "
              f"--init doesn't support (pre-v1.0.13, before the current release format):")
        for tag in unsupported:
            print(f"  {tag}")

    if not missing:
        print("\nNothing to download - every supported release is already downloaded.")
        return

    print(f"\nMissing ({len(missing)}):")
    for tag in missing:
        print(f"  {tag}")

    if args.dry_run:
        print("\n(dry run - not downloading)")
        return

    python_exe = find_python_interpreter()
    print()
    succeeded, failed = [], []
    for i, tag in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] Downloading {tag}...")
        result = subprocess.run([python_exe, "-m", "compete", "--init", "rusty", tag], cwd=REPO_ROOT)
        (succeeded if result.returncode == 0 else failed).append(tag)

    print(f"\n{'='*50}")
    print(f"Downloaded: {len(succeeded)}/{len(missing)}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
    print(f"{'='*50}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
