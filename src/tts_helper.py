import threading
import subprocess
import time

class TextToSpeechEngine:
    def __init__(self, cooldown=3.0):
        self.cooldown = cooldown
        self.last_spoken_time = 0
        self.last_spoken_word = ""
        self._lock = threading.Lock()
        self.enabled = True

    def speak(self, text):
        if not self.enabled:
            return
            
        now = time.time()
        with self._lock:
            # Avoid repeating the same word within cooldown window
            if text == self.last_spoken_word and (now - self.last_spoken_time) < self.cooldown:
                return
            if (now - self.last_spoken_time) < 1.5:
                return
            self.last_spoken_time = now
            self.last_spoken_word = text
            
        def _worker():
            try:
                # Use Windows SAPI speech engine
                safe_text = text.replace('"', '')
                cmd = f'(New-Object -ComObject SAPI.SpVoice).Speak("{safe_text}")'
                subprocess.run(['powershell', '-Command', cmd], capture_output=True, timeout=5)
            except Exception:
                pass
                
        threading.Thread(target=_worker, daemon=True).start()

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
