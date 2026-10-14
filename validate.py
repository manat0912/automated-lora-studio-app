import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Pinokio manages PATH, Python, and GPU runtime for this verification.
subprocess.run([sys.executable,'-m','compileall','-q','studio','app.py','setup.py'],check=True)
subprocess.run([sys.executable,'-m','pytest','tests','-q','--tb=short'],check=True)
print('STUDIO_VALIDATION_PASSED',flush=True)
