# Remote Session Management

`remote_tmux.py` wraps tmux to keep training processes alive across SSH
disconnects and let multiple observers (humans and Claudes) watch output
without interfering with each other.

## Why

Rented GPU nodes die when your SSH connection drops. tmux keeps processes
running in detached sessions on the server side. This script adds:

- Namespaced sessions (`fd_` prefix) so training sessions don't collide
  with anything else on the machine
- Automatic logfile capture via `pipe-pane` so you can read output without
  attaching to tmux at all
- Read-only `watch` mode for observers who should see but not touch
- A `setup` command that launches the standard server/train/sync trio
  with port-readiness polling between server and train startup

## Quick start

### Launch everything at once

```bash
python remote_tmux.py setup \
    --server-args "python -m futudiffu.server --port 5555 --fp8-diff /weights/z_image_fp8.safetensors --te /weights/qwen_3_4b.safetensors --vae /weights/zimage.safetensors" \
    --train-args "python train.py --port 5555 --dataset-dir /scratch/btrm_dataset --output-dir /scratch/run_001" \
    --sync-cmd 'while true; do rsync -avz /scratch/run_001/ user@home:~/runs/001/; sleep 30; done' \
    --port 5555
```

This launches `server` first, waits until port 5555 accepts connections,
then launches `train` and `sync`. All three sessions get logfiles at
`~/.futudiffu_logs/{server,train,sync}.log`.

### Or launch individually

```bash
python remote_tmux.py launch server "python -m futudiffu.server --port 5555 ..."
python remote_tmux.py launch train  "python train.py --port 5555 ..."
python remote_tmux.py launch sync   "while true; do rsync ...; sleep 30; done"
```

## Observing a run

### From a terminal (interactive)

```bash
python remote_tmux.py watch train    # read-only tmux attach (can't send input)
python remote_tmux.py attach train   # full control (careful!)
```

`watch` is safe for anyone to use — it attaches with `-r` (read-only).
You see the live terminal output but keystrokes are ignored.

### From a script or Claude agent (non-interactive)

```bash
python remote_tmux.py logs train     # tail -f the logfile, no tmux needed
```

This works from any SSH session, even one that isn't inside tmux. Useful
for Claude agents that need to read training progress from a plain shell.

## Checking status

```bash
python remote_tmux.py list
```

```
Name                 Uptime         Command
------------------------------------------------------------
server               1h23m45s       python
train                0h47m12s       python
sync                 1h23m40s       bash
```

## Tearing down

```bash
python remote_tmux.py kill train           # prompts for confirmation
python remote_tmux.py kill train --force   # no prompt
```

## Session lifecycle

```
SSH in → setup → (disconnect safely) → SSH back in → watch/logs → kill
                     ↑                                    ↑
              process keeps running              multiple observers
              in detached tmux                   can watch simultaneously
```

The whole point: the training run's lifetime is decoupled from your SSH
session's lifetime. Network blips, laptop lid closes, terminal crashes —
none of these stop the training.

## Logfiles

All session output is captured to `~/.futudiffu_logs/<session-name>.log`
via tmux's `pipe-pane`. These persist even after the session is killed,
so you can inspect post-mortem output.
