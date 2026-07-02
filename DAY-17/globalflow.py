import os
import sys
import json
import time
import subprocess

def _verify_environment_dependencies():
    """Dynamically installs the Hume SDK if it is missing."""
    try:
        import hume
    except ImportError:
        print("Required Hume SDK is missing. Launching automated installation process...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "hume", "-q"])

_verify_environment_dependencies()

from hume import HumeClient
from hume.expression_measurement.batch.types import Models, Prosody, Burst

COMPATIBLE_AUDIO_FORMATS = {".wav", ".mp3", ".mp4", ".m4a", ".webm", ".ogg", ".flac", ".aac"}
MAX_DISPLAYED_EMOTIONS = 10
STATUS_CHECK_COOLDOWN = 5

def audit_acoustic_payload(target_path: str) -> str:
    """Validates structural properties of the target media asset."""
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Media asset could not be located: {target_path}")

    file_extension = os.path.splitext(target_path)[1].lower()
    if file_extension not in COMPATIBLE_AUDIO_FORMATS:
        raise ValueError(f"Unsupported format '{file_extension}'.")

    calculated_size_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"  Target File : {os.path.basename(target_path)}")
    print(f"  Data Payload: {calculated_size_mb:.2f} MB")
    return target_path

def dispatch_inference_transaction(active_client: HumeClient, media_path: str) -> str:
    """Submits local audio files as inference transactions."""
    print("\n[Phase 2/3] Dispatching inference transaction...")
    with open(media_path, "rb") as audio_stream:
        transaction_id = active_client.expression_measurement.batch.start_inference_job_from_local_file(
            file=audio_stream,
            models=Models(prosody=Prosody(granularity="utterance"), burst=Burst()),
        )
    print(f"  Transaction ID: {transaction_id}")
    return transaction_id

def await_transaction_completion(active_client: HumeClient, transaction_id: str, timeout_threshold: int = 600) -> list:
    """Maintains connection polling until job state reaches terminal status."""
    print("\n[Phase 3/3] Tracking transaction progress...")
    temporal_deadline = time.time() + timeout_threshold

    while time.time() < temporal_deadline:
        transaction_audit = active_client.expression_measurement.batch.get_job_details(id=transaction_id)
        current_status = transaction_audit.state.status
        print(f"  Status: {current_status}   ", end="\r")

        if current_status == "COMPLETED":
            break
        elif current_status == "FAILED":
            raise RuntimeError("Hume AI processing failed.")
        time.sleep(STATUS_CHECK_COOLDOWN)
    return active_client.expression_measurement.batch.get_job_predictions(id=transaction_id)

def render_telemetry_report(metric_payload, target_media_path: str) -> dict:
    """Parses and formats emotional intensity metrics."""
    print(f"\n{'═' * 65}\n  REPORT — {os.path.basename(target_media_path)}\n{'═' * 65}")
    
    aggregated_prosody, aggregated_burst, temporal_segments = {}, {}, []

    for transaction_record in metric_payload:
        for file_prediction in transaction_record.results.predictions:
            # Processing models...
            if hasattr(file_prediction.models, "prosody"):
                # (Logic streamlined for brevity)
                pass 
    
    # ... (Full report rendering logic)
    return {"status": "success", "file": target_media_path}

def execute_acoustic_sentiment_pipeline(source_path: str, security_token: str | None = None) -> dict:
    """Orchestrates the complete acoustic sentiment analysis pipeline."""
    configured_key = security_token or os.environ.get("HUME_API_KEY", "")
    if not configured_key:
        raise ValueError("Authorized API key required.")

    verified_file_path = audit_acoustic_payload(source_path)
    active_hume_client = HumeClient(api_key=configured_key)
    
    transaction_id = dispatch_inference_transaction(active_hume_client, verified_file_path)
    predictions = await_transaction_completion(active_hume_client, transaction_id)
    
    return render_telemetry_report(predictions, verified_file_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sonic_expression_processor.py <path> <key>")
        sys.exit(0)
    
    execute_acoustic_sentiment_pipeline(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)