import argparse
import threading
from .engine import _prepare

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--id',required=True)
    _prepare(p.parse_args().id,threading.Event())
