#!/usr/bin/env python3
"""
Quick test script to verify logging configuration is working correctly.
Run this after setting up to ensure both terminal and GCP logging work.
"""

import sys
import os
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env
load_dotenv()

from logger_config import setup_logger, log_structured

def test_logging():
    print("=" * 60)
    print("VitaLITy2 Logging Configuration Test")
    print("=" * 60)

    # Initialize logger
    logger = setup_logger(name="vitality2-test", enable_gcp=True)

    print("\n1. Testing standard logging levels...")
    logger.info("✓ INFO level log - this is a test info message")
    logger.warning("⚠️  WARNING level log - this is a test warning")
    logger.error("❌ ERROR level log - this is a test error")

    print("\n2. Testing structured logging (for Socket.io events)...")
    log_structured(
        event_name="test_event",
        data={
            "user_id": "test_user_123",
            "session_id": "test_session_456",
            "action": "button_click",
            "component": "test_button",
            "timestamp": "2024-01-15 10:30:00"
        },
        level="INFO"
    )
    print("✓ Structured log sent")

    print("\n3. Configuration check...")

    # Check if GCP credentials are set
    gcp_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if gcp_creds:
        if os.path.exists(gcp_creds):
            print(f"✓ GCP credentials found: {gcp_creds}")
            print("✓ Logs should be sent to both TERMINAL and GOOGLE CLOUD")
        else:
            print(f"⚠️  GCP credentials path set but file not found: {gcp_creds}")
            print("⚠️  Logs will only appear in TERMINAL")
    else:
        print("⚠️  GOOGLE_APPLICATION_CREDENTIALS not set in .env")
        print("⚠️  Logs will only appear in TERMINAL")
        print("   To enable GCP logging, see: GOOGLE_CLOUD_LOGGING_SETUP.md")

    print("\n4. Verify in Google Cloud Console...")
    print("   If GCP is configured, check logs at:")
    print("   https://console.cloud.google.com/logs")
    print("   Filter by: logName=projects/YOUR_PROJECT/logs/vitality2-test")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    print("\nWhat you should see:")
    print("- Above messages in your TERMINAL ✓")
    if gcp_creds and os.path.exists(gcp_creds):
        print("- Same messages in GCP Logs Explorer (check in 1-2 minutes)")
    else:
        print("- To enable GCP logging, follow: GOOGLE_CLOUD_LOGGING_SETUP.md")
    print("=" * 60)

    # Flush logs to GCP before exiting (important for short-lived scripts)
    print("\nFlushing logs to Google Cloud...")
    import time
    import logging as logging_module

    # Get all handlers and flush them
    for handler in logger.handlers:
        handler.flush()

    # Give background thread time to send logs
    time.sleep(2)
    print("✓ Logs flushed!")

if __name__ == "__main__":
    try:
        test_logging()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
