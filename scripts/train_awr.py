#!/usr/bin/env python3
"""Train frontier-selector policy from offline transition data (AWR).

Workflow:
  1. Collect transitions (Rust):
       cargo run --release -p exploration_rl --example collect_episodes [episodes] [out.ndjson]

  2. Train (this script):
       python scripts/train_awr.py [transitions.ndjson]

  3. Deploy:
       cp models/frontier_selector.onnx <robot>

Algorithm: Advantage-Weighted Regression (AWR) — offline RL.
  - Compute Monte-Carlo returns G_t (episode-aware, gamma=GAMMA).
  - Normalise to advantage A_t = (G_t - mean) / std.
  - AWR weight w_t = clip(exp(A_t / beta), 0, MAX_WEIGHT).
  - Loss = -mean(w_t * log pi(a_t | s_t))
  Beta controls exploitation/exploration:
    small beta → sharp, learns only from best transitions
    large beta → soft, approaches pure behavioural cloning

Network (must match FrontierSelector ONNX inference in exploration_rl/src/lib.rs):
  Input:  "state"   float32 [batch, 28]
  Output: "logits"  float32 [batch, 5]
  Action IDs: 0=Nearest 1=Largest 2=Leftmost 3=Rightmost 4=RandomValid
"""

import sys
import json
import argparse
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
except ImportError:
    sys.exit("Install: pip install torch numpy")

ROOT      = Path(__file__).resolve().parent.parent
DEF_DATA  = ROOT / "checkpoints" / "transitions.ndjson"
DEF_OUT   = ROOT / "models"      / "frontier_selector.onnx"
DEF_CKPT  = ROOT / "checkpoints" / "frontier_selector.pt"

# ── Constants matching exploration_rl/src/lib.rs ──────────────────────────────

STATE_LEN   = 28   # 4 + 8 frontiers × 3
NUM_ACTIONS = 5    # Nearest/Largest/Leftmost/Rightmost/Random
ACTION_NAMES = ["Nearest", "Largest", "Leftmost", "Rightmost", "Random"]

# ── Hyperparameters ───────────────────────────────────────────────────────────

HIDDEN_DIM  = 64
GAMMA       = 0.99
AWR_BETA    = 0.3       # temperature (0.3=sharp, 1.0=soft, 2.0=very soft)
AWR_MAX_W   = 20.0      # max weight clip
EPOCHS      = 100
BATCH_SIZE  = 512
LR          = 3e-4

# ── Model ─────────────────────────────────────────────────────────────────────

class FrontierPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_LEN,  HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits [batch, 5]

# ── Data ──────────────────────────────────────────────────────────────────────

def load_ndjson(path: Path):
    states, actions, rewards, dones = [], [], [], []
    with open(path) as fh:
        for line in fh:
            t = json.loads(line)
            states.append(t["state"])
            actions.append(t["action"])
            rewards.append(t["reward"])
            dones.append(t["done"])
    return (
        np.array(states,  dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(dones,   dtype=bool),
    )


def monte_carlo_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    """Episode-aware MC returns (backward pass, reset at done boundaries)."""
    G = np.zeros_like(rewards)
    g = 0.0
    for i in reversed(range(len(rewards))):
        if dones[i]:
            g = 0.0
        g = rewards[i] + gamma * g
        G[i] = g
    return G

# ── Training ──────────────────────────────────────────────────────────────────

def train(data_path: Path, device: str) -> FrontierPolicy:
    print(f"Loading {data_path} …")
    states, actions, rewards, dones = load_ndjson(data_path)
    n = len(states)

    G = monte_carlo_returns(rewards, dones, GAMMA)
    A = (G - G.mean()) / (G.std() + 1e-8)
    W = np.clip(np.exp(A / AWR_BETA), 0.0, AWR_MAX_W).astype(np.float32)

    counts = np.bincount(actions, minlength=NUM_ACTIONS)
    print(f"  {n:,} transitions  reward_mean={rewards.mean():.4f}  return_mean={G.mean():.4f}")
    print("  action distribution:")
    for i, (name, c) in enumerate(zip(ACTION_NAMES, counts)):
        print(f"    {i} {name:10s}: {c:6d}  ({100*c/n:.1f}%)")
    print()

    S   = torch.from_numpy(states).to(device)
    A_t = torch.from_numpy(actions).to(device)
    W_t = torch.from_numpy(W).to(device)

    model = FrontierPolicy().to(device)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    idx   = np.arange(n)

    print(f"Training {EPOCHS} epochs on {device} …")
    for epoch in range(1, EPOCHS + 1):
        np.random.shuffle(idx)
        total_loss = 0.0
        batches    = 0
        for start in range(0, n, BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            logits   = model(S[b])
            log_prob = torch.log_softmax(logits, dim=-1)
            log_pa   = log_prob[torch.arange(len(b), device=device), A_t[b]]
            loss     = -(W_t[b] * log_pa).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            batches    += 1
        sched.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            with torch.no_grad():
                pred = model(S).argmax(dim=-1)
            acc = (pred == A_t).float().mean().item()
            print(f"  epoch {epoch:3d}/{EPOCHS}  loss={total_loss/batches:.4f}  "
                  f"acc={acc:.3f}")

    DEF_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DEF_CKPT)
    print(f"\nCheckpoint → {DEF_CKPT}")
    return model

# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model: FrontierPolicy, out: Path) -> None:
    model.eval().cpu()
    dummy = torch.zeros(1, STATE_LEN)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out),
        opset_version=17,
        input_names=["state"],
        output_names=["logits"],
        dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    size_kb = out.stat().st_size / 1024
    print(f"ONNX → {out}  ({size_kb:.1f} KB)")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        out_val = sess.run(None, {"state": dummy.numpy()})[0][0]
        best = int(out_val.argmax())
        print(f"Sanity check: logits={out_val.round(3)}  argmax={best} ({ACTION_NAMES[best]})")
    except ImportError:
        print("(onnxruntime not installed — skipping sanity check)")

# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="AWR trainer for frontier-selector policy")
    p.add_argument("data",  nargs="?", default=str(DEF_DATA), help="NDJSON transitions file")
    p.add_argument("--out", default=str(DEF_OUT),             help="ONNX output path")
    p.add_argument("--cpu", action="store_true",              help="Force CPU training")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(
            f"No data at {data_path}\n"
            "Collect first:\n"
            "  cargo run --release -p exploration_rl --example collect_episodes"
        )

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = train(data_path, device)
    export_onnx(model, Path(args.out))
    print("\nDone — deploy models/frontier_selector.onnx to the robot.")

if __name__ == "__main__":
    main()
