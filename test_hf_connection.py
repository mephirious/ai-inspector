#!/usr/bin/env python3
"""Test script to verify HuggingFace connection and token."""

import os
from pathlib import Path

# Try to load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print("✓ Loaded .env file")
    else:
        print("⚠ .env file not found")
except ImportError:
    print("⚠ python-dotenv not installed, skipping .env loading")

# Check token
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    print(f"✓ Token found (length: {len(token)}, starts with: {token[:10]}...)")
else:
    print("✗ No token found. Set HF_TOKEN in .env or as environment variable")

# Test connection
try:
    from huggingface_hub import list_repo_files, whoami
    
    # Check authentication
    try:
        user_info = whoami(token=token)
        print(f"✓ Authenticated as: {user_info.get('name', 'Unknown')}")
    except Exception as e:
        print(f"⚠ Authentication check failed: {e}")
    
    # Try to list files in the repo
    repo_id = "obazl/yolov8-signature-detection"
    print(f"\nTesting connection to: {repo_id}")
    
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="model", token=token)
        pt_files = [f for f in files if f.endswith(".pt")]
        print(f"✓ Successfully connected! Found {len(files)} files, {len(pt_files)} .pt files")
        if pt_files:
            print(f"  Model files found: {pt_files[:5]}")  # Show first 5
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "gated" in error_str.lower():
            print(f"✗ Authentication error: {error_str}")
            print("  → The model is gated. Make sure:")
            print("    1. You have requested access at https://huggingface.co/tech4humans/yolov8s-signature-detector")
            print("    2. Your token has read access")
            print("    3. Your token is valid")
        elif "connection" in error_str.lower() or "network" in error_str.lower():
            print(f"✗ Connection error: {error_str}")
            print("  → Check your internet connection")
        else:
            print(f"✗ Error: {error_str}")

except ImportError:
    print("✗ huggingface_hub not installed")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

