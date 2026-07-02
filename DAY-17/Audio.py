import os
import sys
import json
import time
import subprocess


# ── Dependency Auto-Verification ─────────────────────────────────────────────
def _verify_environment_dependencies():
    """Dynamically installs the Hume SDK if it is missing in the current system context."""
    try:
        import hume  # noqa: F401
    except ImportError:
        print("Required Hume SDK is missing. Launching automated installation process...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "hume", "-q"])

_verify_environment_dependencies()

from hume import HumeClient                                      # noqa: E402
from hume.expression_measurement.batch.types import (
    Models,
    Prosody,
    Burst,
)


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
COMPATIBLE_AUDIO_FORMATS = {".wav", ".mp3", ".mp4", ".m4a", ".webm", ".ogg", ".flac", ".aac"}
MAX_DISPLAYED_EMOTIONS = 10  # Maximum number of emotions highlighted in console prints
STATUS_CHECK_COOLDOWN = 5   # Interval duration (in seconds) between backend polling calls


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — Payload Verification
# ─────────────────────────────────────────────────────────────────────────────
def audit_acoustic_payload(target_path: str) -> str:
    """
    Validates structural properties of the target audio file prior to processing.
    
    Args:
        target_path: Path pointing to the target media file.
        
    Returns:
        The verified absolute path of the confirmed media asset.
    """
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Media asset could not be located: {target_path}")

    file_extension = os.path.splitext(target_path)[1].lower()
    if file_extension not in COMPATIBLE_AUDIO_FORMATS:
        raise ValueError(
            f"Unsupported file format '{file_extension}'.\n"
            f"Compatible extensions: {', '.join(sorted(COMPATIBLE_AUDIO_FORMATS))}"
        )

    calculated_size_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"  Target File : {os.path.basename(target_path)}")
    print(f"  Data Payload: {calculated_size_mb:.2f} MB")
    print(f"  Asset Type  : {file_extension}")
    return target_path


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — Batch Job Submission
# ─────────────────────────────────────────────────────────────────────────────
def dispatch_inference_transaction(active_client: HumeClient, media_path: str) -> str:
    """
    Submits a raw local audio file as a processing transaction to the Hume AI interface.
    
    Args:
        active_client: Authenticated instance of the Hume API Client.
        media_path: Absolute system path of the validated audio segment.
        
    Returns:
        The unique batch transaction string identifier.
    """
    print("\n[Phase 2/3] Dispatching inference transaction to Hume AI engine...")

    with open(media_path, "rb") as audio_stream:
        transaction_id = active_client.expression_measurement.batch.start_inference_job_from_local_file(
            file=audio_stream,
            models=Models(
                prosody=Prosody(granularity="utterance"),
                burst=Burst(),
            ),
        )

    print(f"  Transaction ID Registered : {transaction_id}")
    return transaction_id


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — Transaction Tracking & Polling
# ─────────────────────────────────────────────────────────────────────────────
def await_transaction_completion(active_client: HumeClient, transaction_id: str, timeout_threshold: int = 600) -> list:
    """
    Maintains a polling connection to track job execution state until completion.
    
    Args:
        active_client: Authenticated instance of the Hume API Client.
        transaction_id: Structural string key associated with the processing job.
        timeout_threshold: Maximum runtime seconds to monitor execution before throwing an exception.
        
    Returns:
        The generated prediction records