import os
import runpy
import sys
repo,config = sys.argv[1:3]
os.chdir(repo)
sys.path.insert(0,repo)
sys.argv=['run.py',config]
runpy.run_path('run.py',run_name='__main__')
