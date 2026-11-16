"""Non-blocking text-to-speech using the operating system's built-in voice.

Windows: SAPI via PowerShell. macOS: `say`. Linux: `spd-say`, `espeak-ng` or `espeak`.
"""
import platform
import re
import shutil
import subprocess
import threading
import time


def _speech_command(text):
    system = platform.system()
    if system == 'Windows':
        return ['powershell', '-NoProfile', '-Command', f'(New-Object -ComObject SAPI.SpVoice).Speak("{text}")']
    if system == 'Darwin' and shutil.which('say'):
        return ['say', text]
    for tool in ('spd-say', 'espeak-ng', 'espeak'):
        if shutil.which(tool):
            return [tool, text]
    return None


class TextToSpeechEngine:
    def __init__(self, cooldown=3.0):
        self.cooldown = cooldown
        self.last_spoken_time = 0
        self.last_spoken_word = ""
        self._lock = threading.Lock()
        self.enabled = True
        self.available = _speech_command('test') is not None
        if not self.available:
            print('Text-to-speech: no system voice found (install espeak-ng on Linux). Speech disabled.')

    def speak(self, text):
        if not self.enabled or not self.available:
            return
        safe_text = re.sub(r'[^A-Za-z0-9 ]', '', text)
        now = time.time()
        with self._lock:
            # Avoid repeating the same word within the cooldown window
            if safe_text == self.last_spoken_word and (now - self.last_spoken_time) < self.cooldown:
                return
            if (now - self.last_spoken_time) < 1.5:
                return
            self.last_spoken_time = now
            self.last_spoken_word = safe_text

        def _worker():
            try:
                subprocess.run(_speech_command(safe_text), capture_output=True, timeout=5)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
