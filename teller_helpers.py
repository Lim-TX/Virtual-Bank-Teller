import json
import os
import glob


def write_teller_state(state_file: str, teller_state: str, latest_audio: str = "") -> None:
    import time
    tmp = state_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"teller_state": teller_state, "latest_audio": latest_audio}, f)
    for attempt in range(10):
        try:
            os.replace(tmp, state_file)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.01)


def cleanup_old_tts_files(static_dir: str, keep: int = 3) -> None:
    pattern = os.path.join(static_dir, "response_*.mp3")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    for old_file in files[:-keep] if keep > 0 else files:
        try:
            os.remove(old_file)
        except OSError:
            pass


def is_debug_mode() -> bool:
    return os.environ.get("TELLER_DEBUG", "0") == "1"
