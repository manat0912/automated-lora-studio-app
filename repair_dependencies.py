import subprocess
import sys
subprocess.run(['uv','pip','install','--python',sys.executable,'-r','requirements.txt','-c','runtime-constraints.txt'],check=True)
subprocess.run(['uv','pip','check','--python',sys.executable],check=True)
