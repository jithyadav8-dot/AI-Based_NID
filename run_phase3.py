# run_phase3.py

import sys

if __name__ == "__main__":
    # Ensure Windows console doesn't crash on unicode characters like █ and ×
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        
    from training.autoencoder import run
    run()
