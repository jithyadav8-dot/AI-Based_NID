# Main entry point for Phase 1
# run_phase1.py

from data.pipeline import run

if __name__ == "__main__":
    run(
        raw_2018_dir = "raw/cic2018",
        out_dir      = "processed",
    )