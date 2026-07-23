import time
import re

log_path = r"C:\Users\Jagajith Yadav\.gemini\antigravity-ide\brain\d996d307-5a77-40e8-963a-32f2f0a85319\.system_generated\tasks\task-236.log"

while True:
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Look for scored=50 or higher
            if re.search(r"scored=([5-9][0-9]|[1-9][0-9]{2,})", content):
                print("Found scored=50+ in log!")
                break
    except Exception:
        pass
    time.sleep(2)
