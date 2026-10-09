"""Start training in a persistent tmux window, retaining its interactive shell."""
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parents[1]
session = 'turbovla-eovsam'
subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-n', 'training', '-c', str(repo)], check=True)
subprocess.run(['tmux', 'set-window-option', '-t', session + ':0', 'remain-on-exit', 'on'], check=True)
subprocess.run(['tmux', 'send-keys', '-t', session + ':0', '-l', 'bash scripts/run_libero_tmux.sh'], check=True)
subprocess.run(['tmux', 'send-keys', '-t', session + ':0', 'Enter'], check=True)
print('tmux attach -t ' + session)
