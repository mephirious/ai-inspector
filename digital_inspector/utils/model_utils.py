"""Utilities for downloading and loading models from HuggingFace."""

import os
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files, login

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip .env loading


def get_model_path(hf_repo: str, filename: Optional[str] = None, token: Optional[str] = None) -> str:
    """
    Download model from HuggingFace and return local path.
    
    Args:
        hf_repo: HuggingFace repository ID (e.g., "tech4humans/yolov8s-signature-detector")
        filename: Specific file to download (e.g., "best.pt"). If None, tries common names.
        token: HuggingFace token for gated repositories. If None, uses HF_TOKEN env var or cached token.
    
    Returns:
        Local path to the model file
    """
    # Get token from parameter, environment variable, or cached token
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Debug: Check if token is available (but don't print the actual token)
    if token:
        print(f"Using HuggingFace token (length: {len(token)})")
    else:
        print("Warning: No HuggingFace token found. If the model is gated, authentication may fail.")
        print("Set HF_TOKEN in .env file or as environment variable.")
    
    # Cache directory for models
    cache_dir = Path.home() / ".cache" / "digital_inspector" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists locally
    repo_name = hf_repo.replace("/", "_")
    local_repo_dir = cache_dir / repo_name
    
    # Try to find model file in cache first
    if local_repo_dir.exists():
        # Look for common model file names
        model_files = list(local_repo_dir.rglob("*.pt"))
        if model_files:
            return str(model_files[0])
    
    # Download from HuggingFace
    print(f"Downloading model from HuggingFace: {hf_repo}")
    
    # Prepare download kwargs (removed deprecated local_dir_use_symlinks)
    download_kwargs = {
        "repo_id": hf_repo,
        "cache_dir": str(cache_dir),
        "local_dir": str(local_repo_dir),
    }
    if token:
        download_kwargs["token"] = token
    
    try:
        # First, try to list files in the repo to find the model file
        try:
            list_kwargs = {"repo_id": hf_repo, "repo_type": "model"}
            if token:
                list_kwargs["token"] = token
            repo_files = list_repo_files(**list_kwargs)
            pt_files = [f for f in repo_files if f.endswith(".pt")]
            
            if pt_files:
                # Prefer best.pt or weights/best.pt, otherwise use first found
                preferred_names = ["best.pt", "weights/best.pt", "train/weights/best.pt"]
                model_filename = None
                for preferred in preferred_names:
                    if preferred in pt_files:
                        model_filename = preferred
                        break
                
                if not model_filename:
                    model_filename = pt_files[0]
                
                print(f"Found model file: {model_filename}")
                
                # Try to download just the specific file first (without local_dir for better reliability)
                try:
                    file_download_kwargs = {
                        "repo_id": hf_repo,
                        "filename": model_filename,
                        "cache_dir": str(cache_dir),
                        "resume_download": True,
                        # Don't use local_dir for single file downloads - it can cause issues
                    }
                    if token:
                        file_download_kwargs["token"] = token
                    
                    print(f"Attempting to download: {model_filename}")
                    model_path = hf_hub_download(**file_download_kwargs)
                    
                    # Verify the file exists
                    if os.path.exists(model_path):
                        print(f"Successfully downloaded model file: {model_path}")
                        # Copy to our local_dir for consistency
                        local_file_path = local_repo_dir / model_filename
                        local_file_path.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy2(model_path, local_file_path)
                        return str(local_file_path)
                    else:
                        print(f"Warning: Download reported success but file not found at: {model_path}")
                except Exception as file_error:
                    error_str = str(file_error)
                    if "401" in error_str or "gated" in error_str.lower() or "restricted" in error_str.lower():
                        raise RuntimeError(
                            f"Model {hf_repo} requires authentication.\n"
                            f"Please set HF_TOKEN in .env file or as environment variable.\n"
                            f"Get token from: https://huggingface.co/settings/tokens"
                        )
                    print(f"Direct file download failed: {error_str}")
                    print("Trying alternative download methods...")
                    # Fall through to repository download
        except Exception as e:
            if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
                raise RuntimeError(
                    f"Model {hf_repo} is gated/restricted and requires authentication.\n"
                    f"Please:\n"
                    f"1. Request access at https://huggingface.co/{hf_repo}\n"
                    f"2. Get your HuggingFace token from https://huggingface.co/settings/tokens\n"
                    f"3. Set it as environment variable: export HF_TOKEN=your_token_here\n"
                    f"   Or login: huggingface-cli login\n"
                    f"   Or pass it to the detector: SignatureDetector(token='your_token')"
                )
            print(f"File listing failed: {e}, trying repository download...")
        
        # Fallback: download entire repo (but only if we haven't found the file yet)
        print("Attempting to download entire repository...")
        try:
            # Add resume_download for better reliability
            snapshot_kwargs = download_kwargs.copy()
            snapshot_kwargs["resume_download"] = True
            snapshot_download(**snapshot_kwargs)
            print("Repository download completed")
        except Exception as snapshot_error:
            error_str = str(snapshot_error)
            if "401" in error_str or "gated" in error_str.lower() or "restricted" in error_str.lower():
                raise RuntimeError(
                    f"Model {hf_repo} requires authentication.\n"
                    f"Please set HF_TOKEN in .env file or as environment variable.\n"
                    f"Get token from: https://huggingface.co/settings/tokens"
                )
            # If snapshot download fails, try common filenames directly
            print(f"Repository download failed: {error_str}")
            print("Trying to download common model filenames directly...")
        
        # Find .pt file in downloaded directory
        model_files = list(local_repo_dir.rglob("*.pt"))
        if model_files:
            return str(model_files[0])
        
        # Try common filenames as last resort (without local_dir)
        common_names = ["best.pt", "weights/best.pt", "train/weights/best.pt", "model.pt", "yolov8s.pt", "yolov8n.pt"]
        for common_name in common_names:
            try:
                print(f"Trying to download: {common_name}")
                file_download_kwargs = {
                    "repo_id": hf_repo,
                    "filename": common_name,
                    "cache_dir": str(cache_dir),
                    "resume_download": True,
                    # Don't use local_dir for single file downloads
                }
                if token:
                    file_download_kwargs["token"] = token
                
                model_path = hf_hub_download(**file_download_kwargs)
                if os.path.exists(model_path):
                    print(f"Successfully downloaded: {common_name}")
                    # Copy to our local_dir for consistency
                    local_file_path = local_repo_dir / common_name
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(model_path, local_file_path)
                    return str(local_file_path)
            except Exception as e:
                error_str = str(e)
                if "401" in error_str or "gated" in error_str.lower():
                    raise RuntimeError(
                        f"Model {hf_repo} requires authentication.\n"
                        f"Please set HF_TOKEN in .env file or as environment variable.\n"
                        f"Get token from: https://huggingface.co/settings/tokens"
                    )
                print(f"Failed to download {common_name}: {error_str}")
                continue
        
        raise FileNotFoundError(
            f"No .pt model file found in {hf_repo}. "
            f"Please check the repository structure or specify the filename."
        )
    
    except RuntimeError:
        # Re-raise our custom authentication errors
        raise
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "gated" in error_msg.lower() or "restricted" in error_msg.lower():
            raise RuntimeError(
                f"Model {hf_repo} is gated/restricted and requires authentication.\n"
                f"Please:\n"
                f"1. Request access at https://huggingface.co/{hf_repo}\n"
                f"2. Get your HuggingFace token from https://huggingface.co/settings/tokens\n"
                f"3. Set it as environment variable: export HF_TOKEN=your_token_here\n"
                f"   Or login: huggingface-cli login\n"
                f"   Or pass it to the detector: SignatureDetector(token='your_token')"
            )
        raise RuntimeError(f"Failed to download model from {hf_repo}: {error_msg}")

