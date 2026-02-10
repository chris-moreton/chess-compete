"""
S3 build cache utilities for sharing compiled engine binaries between workers.

All functions are defensive — S3 failures silently fall through to local builds.
Uses the aws CLI via subprocess (available on Amazon Linux 2023 by default).
"""

import subprocess


def s3_download(bucket: str, key: str, local_path: str) -> bool:
    """
    Download a binary from S3.

    Returns True if found and downloaded successfully, False on any failure.
    """
    try:
        result = subprocess.run(
            ['aws', 's3', 'cp', f's3://{bucket}/{key}', local_path, '--quiet'],
            capture_output=True, timeout=60
        )
        return result.returncode == 0
    except Exception:
        return False


def s3_upload(bucket: str, key: str, local_path: str) -> bool:
    """
    Upload a binary to S3.

    Returns True on success. Fire-and-forget — failure just means the next
    worker will build too.
    """
    try:
        result = subprocess.run(
            ['aws', 's3', 'cp', local_path, f's3://{bucket}/{key}', '--quiet'],
            capture_output=True, timeout=120
        )
        return result.returncode == 0
    except Exception:
        return False


def s3_exists(bucket: str, key: str) -> bool:
    """
    Check if an object exists in S3 via HEAD request.

    Returns True if the object exists, False otherwise.
    """
    try:
        result = subprocess.run(
            ['aws', 's3api', 'head-object', '--bucket', bucket, '--key', key],
            capture_output=True, timeout=15
        )
        return result.returncode == 0
    except Exception:
        return False


def s3_cleanup(bucket: str, prefix: str) -> bool:
    """
    Delete all objects under a prefix in S3.

    Uses `aws s3 rm --recursive`. Returns True on success.
    """
    try:
        result = subprocess.run(
            ['aws', 's3', 'rm', f's3://{bucket}/{prefix}', '--recursive', '--quiet'],
            capture_output=True, timeout=60
        )
        return result.returncode == 0
    except Exception:
        return False
