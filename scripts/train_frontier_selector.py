#!/usr/bin/env python3
"""
PPO training script for the frontier-selector policy.

Trains a small MLP policy to choose which frontier to navigate to.
The state vector (28 floats) exactly matches the Rust build_state_vector()
in exploration_rl/src/lib.rs:
  [x/10, y/10, cos(theta), sin(theta),
   frontier_0_dx/5, frontier_0_dy/5, frontier_0_size/1000,
   ... (8 frontiers total, zero-padded)]

Output: models/frontier_selector.onnx (dropped in project root's models/)

Usage:
  cd /home/mike/Projects/Robot
  python scripts/train_frontier_selector.py [--episodes N] [--steps N] [--lr F]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# ── Section 0: GPU check and speed benchmark ──────────────────────────────────

def check_gpu() -> "torch.device":
    """Verify the RTX 3050 (not Intel iGPU) is being used, then benchmark it."""
    import torch

    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        print("  Possible causes:")
        print("  - nvidia driver not loaded  (run: nvidia-smi)")
        print("  - PyTorch CPU-only build    (reinstall with CUDA)")
        print("  - No NVIDIA GPU detected")
        sys.exit(1)

    n = torch.cuda.device_count()
    print(f"CUDA devices found: {n}")
    rtx_idx = None
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        print(f"  [{i}] {name}  —  {vram_gb:.1f} GB VRAM")
        if "3050" in name or "RTX" in name.upper():
            if rtx_idx is None:
                rtx_idx = i

    if rtx_idx is None:
        print()
        print("WARNING: No RTX 3050 detected among CUDA devices.")
        print("  If you expected to use the RTX 3050, check:")
        print("  1. nvidia-smi  (driver loaded?)")
        print("  2. /etc/X11/xorg.conf  (PrimaryGPU setting)")
        print("  3. sudo prime-select nvidia  (if optimus system)")
        print("Continuing on device 0...")
        rtx_idx = 0
    else:
        print(f"\nUsing device [{rtx_idx}]: {torch.cuda.get_device_name(rtx_idx)}")

    device = torch.device(f"cuda:{rtx_idx}")
    torch.cuda.set_device(device)

    # ── Speed benchmark ────────────────────────────────────────────────────────
    print()
    print("SPEED BENCHMARK  (4096 × 4096 float32 matmul, 50 iterations)")
    SIZE = 4096
    ITERS = 50

    # CPU
    a_cpu = torch.randn(SIZE, SIZE)
    b_cpu = torch.randn(SIZE, SIZE)
    # warmup
    _ = a_cpu @ b_cpu
    t0 = time.perf_counter()
    for _ in range(ITERS):
        c = a_cpu @ b_cpu
    cpu_ms = (time.perf_counter() - t0) * 1000 / ITERS
    # GFLOPS: 2*N^3 FLOPs per matmul
    cpu_gflops = 2 * SIZE**3 / (cpu_ms * 1e-3) / 1e9
    print(f"  CPU : {cpu_ms:7.1f} ms/iter  ({cpu_gflops:6.1f} GFLOPS)")

    # GPU
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)
    # warmup + sync
    _ = a_gpu @ b_gpu
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        c = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1000 / ITERS
    gpu_gflops = 2 * SIZE**3 / (gpu_ms * 1e-3) / 1e9
    print(f"  GPU : {gpu_ms:7.1f} ms/iter  ({gpu_gflops:6.1f} GFLOPS)")
    print(f"  Speedup: {cpu_ms / gpu_ms:.1f}x")

    if gpu_ms > cpu_ms:
        print("  WARNING: GPU is slower than CPU — driver or PCIe issue likely.")
    print("=" * 60)
    print()

    return device


# ── Section 1: Python FastSim ─────────────────────────────────────────────────

RESOLUTION_M: float = 0.05
GRID_CELLS:   int   = 200
ARENA_M:      float = GRID_CELLS * RESOLUTION_M   # 10.0 m
ROBOT_RADIUS: float = 0.15                         # m
DT_S:         float = 0.1
NUM_RAYS:     int   = 48
HALF_FOV:     float = 55.0 * np.pi / 180.0
MAX_RANGE_M:  float = 3.0
RAY_STEP_M:   float = 0.01
SPEED_M_S:    float = 0.30
OMEGA_RAD_S:  float = 1.0
MAX_STEPS:    int   = 2_000
NUM_OBSTACLES: int  = 18


def _xorshift64(state: int) -> Tuple[int, int]:
    x = state & 0xFFFF_FFFF_FFFF_FFFF
    x ^= (x << 13) & 0xFFFF_FFFF_FFFF_FFFF
    x ^= (x >> 7)
    x ^= (x << 17) & 0xFFFF_FFFF_FFFF_FFFF
    return x, x


class Rng:
    """XorShift64 matching sim_fast/src/lib.rs."""

    def __init__(self, seed: int) -> None:
        self._s = seed if seed != 0 else 0xDEADBEEF_CAFEBABE

    def next_u64(self) -> int:
        self._s, v = _xorshift64(self._s)
        return v

    def next_usize(self, n: int) -> int:
        return self.next_u64() % n

    def next_f32(self) -> float:
        return (self.next_u64() >> 11) / (1 << 53)


class FastSim:
    """
    2-D simulator matching sim_fast/src/lib.rs.

    All raycasting is vectorised over the 48 rays using NumPy.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = Rng(seed)
        self._grid: np.ndarray = np.zeros((GRID_CELLS, GRID_CELLS), dtype=bool)
        self.rx: float = ARENA_M * 0.5
        self.ry: float = ARENA_M * 0.5
        self.rtheta: float = 0.0
        self.step_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        self.step_count = 0
        self._generate_maze()
        return self._make_obs(collision=False, done=False)

    def step(self, action: int) -> dict:
        vx, vy, omega = _action_to_vel(action)
        cos_t = np.cos(self.rtheta)
        sin_t = np.sin(self.rtheta)
        try_x = self.rx + (vx * cos_t - vy * sin_t) * DT_S
        try_y = self.ry + (vx * sin_t + vy * cos_t) * DT_S
        new_theta = _wrap_angle(self.rtheta + omega * DT_S)

        collision = not self._robot_clear(try_x, try_y)
        if not collision:
            self.rx = try_x
            self.ry = try_y
        self.rtheta = new_theta
        self.step_count += 1

        done = collision or self.step_count >= MAX_STEPS
        return self._make_obs(collision=collision, done=done)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _make_obs(self, collision: bool, done: bool) -> dict:
        return {
            "pose": (self.rx, self.ry, self.rtheta),
            "ranges": self._cast_lidar(),
            "collision": collision,
            "done": done,
        }

    def _cast_lidar(self) -> np.ndarray:
        """Vectorised raycasting over NUM_RAYS rays."""
        fracs = np.linspace(0, 1, NUM_RAYS)
        local_angles = -HALF_FOV + fracs * 2 * HALF_FOV
        world_angles = self.rtheta + local_angles

        # Steps along each ray: shape (NUM_RAYS, steps)
        steps = np.arange(RAY_STEP_M, MAX_RANGE_M + RAY_STEP_M, RAY_STEP_M)
        # xs[r, s] = self.rx + steps[s] * cos(world_angles[r])
        xs = self.rx + np.outer(np.cos(world_angles), steps)  # (R, S)
        ys = self.ry + np.outer(np.sin(world_angles), steps)  # (R, S)

        cxs = (xs / RESOLUTION_M).astype(np.int32)
        cys = (ys / RESOLUTION_M).astype(np.int32)

        # Out-of-bounds → treated as wall
        oob = (cxs < 0) | (cxs >= GRID_CELLS) | (cys < 0) | (cys >= GRID_CELLS)
        cxs_clipped = np.clip(cxs, 0, GRID_CELLS - 1)
        cys_clipped = np.clip(cys, 0, GRID_CELLS - 1)
        hit_wall = oob | self._grid[cys_clipped, cxs_clipped]

        # First hit along each ray
        ranges = np.full(NUM_RAYS, MAX_RANGE_M)
        for r in range(NUM_RAYS):
            hits = np.where(hit_wall[r])[0]
            if hits.size > 0:
                ranges[r] = steps[hits[0]]
        return ranges

    def _robot_clear(self, x: float, y: float) -> bool:
        r = ROBOT_RADIUS
        cell_r = int(np.ceil(r / RESOLUTION_M)) + 1
        cx = int(x / RESOLUTION_M)
        cy = int(y / RESOLUTION_M)
        for dx in range(-cell_r, cell_r + 1):
            for dy in range(-cell_r, cell_r + 1):
                nx, ny = cx + dx, cy + dy
                if nx < 0 or ny < 0 or nx >= GRID_CELLS or ny >= GRID_CELLS:
                    return False
                dist = np.hypot(dx * RESOLUTION_M, dy * RESOLUTION_M)
                if dist <= r and self._grid[ny, nx]:
                    return False
        return True

    def _generate_maze(self) -> None:
        g = self._grid
        g[:] = False

        # Border walls (3 cells thick)
        g[:3, :] = True
        g[-3:, :] = True
        g[:, :3] = True
        g[:, -3:] = True

        # Random rectangular obstacles
        min_c, max_c = 4, 30
        for _ in range(NUM_OBSTACLES):
            w = min_c + self._rng.next_usize(max_c - min_c + 1)
            h = min_c + self._rng.next_usize(max_c - min_c + 1)
            x0 = 4 + self._rng.next_usize(GRID_CELLS - w - 8)
            y0 = 4 + self._rng.next_usize(GRID_CELLS - h - 8)
            g[y0:min(y0 + h, GRID_CELLS), x0:min(x0 + w, GRID_CELLS)] = True

        # Place robot
        sx, sy = ARENA_M * 0.5, ARENA_M * 0.5
        self.rx, self.ry = self._nearest_free(sx, sy)
        self.rtheta = self._rng.next_f32() * 2 * np.pi

    def _nearest_free(self, wx: float, wy: float) -> Tuple[float, float]:
        cx0 = int(wx / RESOLUTION_M)
        cy0 = int(wy / RESOLUTION_M)
        for radius in range(GRID_CELLS // 2):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    nx, ny = cx0 + dx, cy0 + dy
                    if nx < 0 or ny < 0 or nx >= GRID_CELLS or ny >= GRID_CELLS:
                        continue
                    px = nx * RESOLUTION_M + RESOLUTION_M * 0.5
                    py = ny * RESOLUTION_M + RESOLUTION_M * 0.5
                    if self._robot_clear(px, py):
                        return px, py
        return wx, wy


def _action_to_vel(action: int) -> Tuple[float, float, float]:
    return [
        (SPEED_M_S,  0.0,          0.0),
        (0.0,        0.0,          OMEGA_RAD_S),
        (0.0,        0.0,         -OMEGA_RAD_S),
        (0.0,        SPEED_M_S,    0.0),
        (0.0,       -SPEED_M_S,    0.0),
        (0.0,        0.0,          0.0),
    ][action]


def _wrap_angle(a: float) -> float:
    a = a % (2 * np.pi)
    if a > np.pi:
        a -= 2 * np.pi
    return a


# ── Section 2: Mapper (lightweight, for state extraction) ─────────────────────

MIN_SIZE_CELLS: int = 4
MAX_FRONTIERS:  int = 8


@dataclass
class Frontier:
    cx: float = 0.0   # centroid x (metres)
    cy: float = 0.0   # centroid y (metres)
    size: int  = 0    # cells


class PyMapper:
    """
    Stripped-down occupancy grid for training.

    Uses the same log-odds constants as mapping/src/lib.rs.
    Computes frontiers via BFS cluster detection.
    """

    LOG_HIT:   float = 0.7
    LOG_MISS:  float = -0.35
    LOG_CLAMP: float = 3.0
    FREE_THRESH: float = -0.5
    OCC_THRESH:  float = 0.5

    def __init__(self) -> None:
        self._logodds: np.ndarray = np.zeros((GRID_CELLS, GRID_CELLS), dtype=np.float32)
        self.explored_cells: int = 0

    def update(self, ranges: np.ndarray, pose: Tuple[float, float, float]) -> List[Frontier]:
        rx, ry, rtheta = pose
        fracs = np.linspace(0, 1, NUM_RAYS)
        local_angles = -HALF_FOV + fracs * 2 * HALF_FOV
        world_angles = rtheta + local_angles

        for i, (angle, rng) in enumerate(zip(world_angles, ranges)):
            hit = rng < MAX_RANGE_M - 1e-3
            end_x = rx + np.cos(angle) * rng
            end_y = ry + np.sin(angle) * rng
            # Bresenham free cells
            self._ray_free(rx, ry, end_x, end_y)
            if hit:
                cx = int(end_x / RESOLUTION_M)
                cy = int(end_y / RESOLUTION_M)
                if 0 <= cx < GRID_CELLS and 0 <= cy < GRID_CELLS:
                    self._logodds[cy, cx] = np.clip(
                        self._logodds[cy, cx] + self.LOG_HIT,
                        -self.LOG_CLAMP, self.LOG_CLAMP
                    )

        self.explored_cells = int(np.sum(self._logodds < self.FREE_THRESH))
        return self._detect_frontiers()

    def _ray_free(self, ox: float, oy: float, ex: float, ey: float) -> None:
        x0 = int(ox / RESOLUTION_M)
        y0 = int(oy / RESOLUTION_M)
        x1 = int(ex / RESOLUTION_M)
        y1 = int(ey / RESOLUTION_M)
        # Simple DDA
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        steps = dx + dy + 1
        for _ in range(steps):
            if 0 <= x < GRID_CELLS and 0 <= y < GRID_CELLS:
                self._logodds[y, x] = np.clip(
                    self._logodds[y, x] + self.LOG_MISS,
                    -self.LOG_CLAMP, self.LOG_CLAMP
                )
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _detect_frontiers(self) -> List[Frontier]:
        free = self._logodds < self.FREE_THRESH
        occ  = self._logodds > self.OCC_THRESH
        unknown = (self._logodds >= self.FREE_THRESH) & (self._logodds <= self.OCC_THRESH)

        # Frontier cells: free cell adjacent to unknown cell
        frontier_mask = np.zeros((GRID_CELLS, GRID_CELLS), dtype=bool)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            shifted_unknown = np.roll(np.roll(unknown, dy, axis=0), dx, axis=1)
            frontier_mask |= (free & shifted_unknown)

        # BFS cluster
        visited = np.zeros((GRID_CELLS, GRID_CELLS), dtype=bool)
        frontiers: List[Frontier] = []
        ys, xs = np.where(frontier_mask)
        for sy, sx in zip(ys, xs):
            if visited[sy, sx]:
                continue
            cluster_x: List[int] = []
            cluster_y: List[int] = []
            queue = collections.deque([(int(sy), int(sx))])
            visited[sy, sx] = True
            while queue:
                cy, cx = queue.popleft()
                cluster_y.append(cy)
                cluster_x.append(cx)
                for ny, nx in [(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)]:
                    if 0 <= ny < GRID_CELLS and 0 <= nx < GRID_CELLS:
                        if frontier_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            if len(cluster_x) >= MIN_SIZE_CELLS:
                frontiers.append(Frontier(
                    cx=float(np.mean(cluster_x)) * RESOLUTION_M,
                    cy=float(np.mean(cluster_y)) * RESOLUTION_M,
                    size=len(cluster_x),
                ))
        return frontiers


# ── Section 3: State vector (must match Rust build_state_vector) ──────────────

STATE_DIM:  int = 4 + MAX_FRONTIERS * 3   # = 28
ACTION_DIM: int = 5


def build_state_vector(
    frontiers: List[Frontier],
    pose: Tuple[float, float, float],
) -> np.ndarray:
    rx, ry, rtheta = pose
    v = [rx / 10.0, ry / 10.0, np.cos(rtheta), np.sin(rtheta)]

    # Sort by distance (nearest first)
    sorted_f = sorted(frontiers, key=lambda f: (f.cx - rx)**2 + (f.cy - ry)**2)

    for i in range(MAX_FRONTIERS):
        if i < len(sorted_f):
            f = sorted_f[i]
            v.extend([
                (f.cx - rx) / 5.0,
                (f.cy - ry) / 5.0,
                f.size / 1000.0,
            ])
        else:
            v.extend([0.0, 0.0, 0.0])

    return np.array(v, dtype=np.float32)


def compute_reward(
    new_explored: int,
    prev_explored: int,
    collision: bool,
    reached_goal: bool,
) -> float:
    return (
        max(new_explored - prev_explored, 0) * 0.01
        + (-5.0 if collision else 0.0)
        + (2.0  if reached_goal else 0.0)
    )


# ── Section 4: RL environment ─────────────────────────────────────────────────

# ── Render helpers (pure numpy, no extra deps) ────────────────────────────────

def _draw_dot(img: np.ndarray, cx: int, cy: int, color: tuple, radius: int) -> None:
    H, W = img.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W:
                    img[ny, nx] = color


def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple) -> None:
    H, W = img.shape[:2]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    for _ in range(dx + dy + 2):
        if 0 <= y < H and 0 <= x < W:
            img[y, x] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _draw_ring(img: np.ndarray, cx: int, cy: int, r: int, color: tuple) -> None:
    H, W = img.shape[:2]
    for angle in np.linspace(0, 2 * np.pi, max(8, r * 4)):
        px = int(cx + r * np.cos(angle))
        py = int(cy + r * np.sin(angle))
        if 0 <= py < H and 0 <= px < W:
            img[py, px] = color


class FrontierEnv:
    """Wraps FastSim + PyMapper into a gym-like environment."""

    def __init__(self, seed: int = 42) -> None:
        self._sim             = FastSim(seed)
        self._mapper          = PyMapper()
        self._last_ranges:    np.ndarray     = np.full(NUM_RAYS, MAX_RANGE_M, dtype=np.float32)
        self._last_frontiers: List[Frontier] = []

    @property
    def explored_cells(self) -> int:
        return self._mapper.explored_cells

    def reset(self) -> np.ndarray:
        obs = self._sim.reset()
        self._mapper         = PyMapper()
        frontiers            = self._mapper.update(obs["ranges"], obs["pose"])
        self._last_ranges    = obs["ranges"]
        self._last_frontiers = frontiers
        self._prev_explored  = self._mapper.explored_cells
        return build_state_vector(frontiers, obs["pose"])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        obs                  = self._sim.step(action)
        frontiers            = self._mapper.update(obs["ranges"], obs["pose"])
        self._last_ranges    = obs["ranges"]
        self._last_frontiers = frontiers
        reward = compute_reward(
            self._mapper.explored_cells,
            self._prev_explored,
            obs["collision"],
            False,
        )
        self._prev_explored = self._mapper.explored_cells
        state = build_state_vector(frontiers, obs["pose"])
        return state, reward, obs["done"]

    def render_map(self, scale: int = 3) -> np.ndarray:
        """Top-down occupancy grid (H×W×3 uint8).
        White=free, dark=occupied, gray=unknown, red=robot, blue=frontiers."""
        lo  = self._mapper._logodds
        img = np.full((GRID_CELLS, GRID_CELLS, 3), 128, dtype=np.uint8)
        img[lo < PyMapper.FREE_THRESH] = (235, 235, 235)
        img[lo > PyMapper.OCC_THRESH]  = (30,  30,  30)
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
        H, W = img.shape[:2]

        for f in self._last_frontiers:
            fx = int(f.cx / RESOLUTION_M) * scale + scale // 2
            fy = int(f.cy / RESOLUTION_M) * scale + scale // 2
            _draw_dot(img, fx, fy, (50, 120, 255), radius=3)

        rx = int(self._sim.rx / RESOLUTION_M) * scale + scale // 2
        ry = int(self._sim.ry / RESOLUTION_M) * scale + scale // 2
        _draw_dot(img, rx, ry, (255, 60, 60), radius=4)
        for t in range(2, 18):
            hx = int(rx + t * np.cos(self._sim.rtheta))
            hy = int(ry + t * np.sin(self._sim.rtheta))
            if 0 <= hy < H and 0 <= hx < W:
                img[hy, hx] = (255, 220, 0)

        return img

    def render_lidar(self, size: int = 300) -> np.ndarray:
        """Polar fan view of the current lidar scan (H×W×3 uint8).
        Forward is up. Red=close, green=far. Rings at 1m, 2m, 3m."""
        img      = np.full((size, size, 3), 15, dtype=np.uint8)
        cx, cy   = size // 2, size // 2
        px_per_m = (size // 2 - 15) / MAX_RANGE_M

        for ring_m in (1.0, 2.0, 3.0):
            _draw_ring(img, cx, cy, int(ring_m * px_per_m), (45, 45, 45))

        fracs        = np.linspace(0, 1, NUM_RAYS)
        local_angles = -HALF_FOV + fracs * 2 * HALF_FOV
        for angle, rng in zip(local_angles, self._last_ranges):
            t     = 1.0 - rng / MAX_RANGE_M        # 0=far(green), 1=close(red)
            color = (int(255 * t), int(255 * (1 - t)), 0)
            # forward=up: ix = cx + r*sin(a),  iy = cy - r*cos(a)
            ex = int(cx + rng * px_per_m * np.sin(angle))
            ey = int(cy - rng * px_per_m * np.cos(angle))
            _draw_line(img, cx, cy, ex, ey, color)
            if 0 <= ey < size and 0 <= ex < size:
                img[ey, ex] = (255, 255, 255)   # endpoint dot

        _draw_dot(img, cx, cy, (255, 60, 60), radius=4)
        for t in range(5, 22):                      # yellow forward tick
            if 0 <= cy - t < size:
                img[cy - t, cx] = (255, 220, 0)

        return img

    def render_combined(self, scale: int = 3) -> np.ndarray:
        """Map + current lidar scan overlaid in world frame.

        Everything is in the same coordinate system as the map, so the ray
        fan matches the robot heading arrow exactly.  Red=close, green=far."""
        img = self.render_map(scale=scale)

        px_per_m = scale / RESOLUTION_M          # pixels per metre in this image
        rx_px = int(self._sim.rx / RESOLUTION_M) * scale + scale // 2
        ry_px = int(self._sim.ry / RESOLUTION_M) * scale + scale // 2

        fracs        = np.linspace(0, 1, NUM_RAYS)
        local_angles = -HALF_FOV + fracs * 2 * HALF_FOV

        for angle, rng in zip(local_angles, self._last_ranges):
            world_angle = self._sim.rtheta + angle
            t     = 1.0 - rng / MAX_RANGE_M     # 0=far(green), 1=close(red)
            color = (int(200 * t), int(200 * (1 - t)), 60)
            ex = int(rx_px + rng * px_per_m * np.cos(world_angle))
            ey = int(ry_px + rng * px_per_m * np.sin(world_angle))
            _draw_line(img, rx_px, ry_px, ex, ey, color)
            if 0 <= ey < img.shape[0] and 0 <= ex < img.shape[1]:
                img[ey, ex] = (255, 255, 255)    # endpoint dot

        # Re-draw robot on top so it isn't obscured by rays
        _draw_dot(img, rx_px, ry_px, (255, 60, 60), radius=4)
        return img


# ── Section 5: Policy and value networks ──────────────────────────────────────

def build_networks(device: "torch.device"):
    import torch
    import torch.nn as nn

    class PolicyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, 64), nn.ReLU(),
                nn.Linear(64, 64),        nn.ReLU(),
                nn.Linear(64, ACTION_DIM),
            )

        def forward(self, x):
            return self.net(x)

    class ValueNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, 64), nn.ReLU(),
                nn.Linear(64, 64),        nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    policy = PolicyNet().to(device)
    value  = ValueNet().to(device)
    return policy, value


# ── Section 6: PPO agent ──────────────────────────────────────────────────────

CLIP_EPS:    float = 0.2
ENT_COEF:    float = 0.05
VF_COEF:     float = 0.5
GAE_LAMBDA:  float = 0.95
GAMMA:       float = 0.99
LR:          float = 3e-4
N_EPOCHS:    int   = 4
BATCH_SIZE:  int   = 64


@dataclass
class Rollout:
    states:   List[np.ndarray] = field(default_factory=list)
    actions:  List[int]        = field(default_factory=list)
    rewards:  List[float]      = field(default_factory=list)
    dones:    List[bool]       = field(default_factory=list)
    log_probs: List[float]     = field(default_factory=list)
    values:   List[float]      = field(default_factory=list)


def collect_rollout(
    env: FrontierEnv,
    policy,
    value_net,
    n_steps: int,
    device,
) -> Rollout:
    import torch
    import torch.nn.functional as F

    rollout = Rollout()
    state = env.reset()

    for _ in range(n_steps):
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = policy(s_t)
            v      = value_net(s_t)
        probs = F.softmax(logits, dim=-1)
        dist  = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        log_p  = dist.log_prob(torch.tensor(action, device=device)).item()

        next_state, reward, done = env.step(action)

        rollout.states.append(state)
        rollout.actions.append(action)
        rollout.rewards.append(reward)
        rollout.dones.append(done)
        rollout.log_probs.append(log_p)
        rollout.values.append(v.item())

        state = next_state if not done else env.reset()

    return rollout


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns    = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + GAMMA * next_val * mask - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * mask * gae
        advantages[t] = gae
        returns[t]    = gae + values[t]
    return advantages, returns


def ppo_update(
    policy,
    value_net,
    optimiser,
    rollout: Rollout,
    advantages: np.ndarray,
    returns: np.ndarray,
    device,
) -> Tuple[float, float, float]:
    import torch
    import torch.nn.functional as F

    states_t     = torch.tensor(np.array(rollout.states), dtype=torch.float32, device=device)
    actions_t    = torch.tensor(rollout.actions, dtype=torch.long,    device=device)
    old_log_p_t  = torch.tensor(rollout.log_probs, dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns_t    = torch.tensor(returns,    dtype=torch.float32, device=device)

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    T = len(rollout.states)
    policy_losses, value_losses, entropies = [], [], []

    for _ in range(N_EPOCHS):
        idx = torch.randperm(T, device=device)
        for start in range(0, T, BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            logits = policy(states_t[b])
            v_pred = value_net(states_t[b])

            dist       = torch.distributions.Categorical(logits=logits)
            new_log_p  = dist.log_prob(actions_t[b])
            entropy    = dist.entropy().mean()

            ratio     = (new_log_p - old_log_p_t[b]).exp()
            surr1     = ratio * advantages_t[b]
            surr2     = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages_t[b]
            pol_loss  = -torch.min(surr1, surr2).mean()
            val_loss  = F.mse_loss(v_pred, returns_t[b])
            loss      = pol_loss + VF_COEF * val_loss - ENT_COEF * entropy

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_net.parameters()), 0.5
            )
            optimiser.step()

            policy_losses.append(pol_loss.item())
            value_losses.append(val_loss.item())
            entropies.append(entropy.item())

    return (
        float(np.mean(policy_losses)),
        float(np.mean(value_losses)),
        float(np.mean(entropies)),
    )


# ── Section 7: Training loop ──────────────────────────────────────────────────

def train(
    n_episodes: int,
    steps_per_episode: int,
    lr: float,
    device,
) -> None:
    import torch

    policy, value_net = build_networks(device)
    optimiser = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=lr,
    )

    env = FrontierEnv(seed=42)

    print(f"Training: {n_episodes} episodes, {steps_per_episode} steps/episode")
    print(f"Device:   {device}")
    print(f"Params:   policy={sum(p.numel() for p in policy.parameters())}, "
          f"value={sum(p.numel() for p in value_net.parameters())}")
    print()

    total_reward = 0.0
    t_start = time.perf_counter()

    for ep in range(1, n_episodes + 1):
        rollout = collect_rollout(env, policy, value_net, steps_per_episode, device)

        # Bootstrap last value
        import torch
        last_s = torch.tensor(rollout.states[-1], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            last_v = value_net(last_s).item()

        advantages, returns = compute_gae(
            rollout.rewards, rollout.values, rollout.dones, last_v
        )

        pol_loss, val_loss, entropy = ppo_update(
            policy, value_net, optimiser, rollout, advantages, returns, device
        )

        ep_reward = sum(rollout.rewards)
        total_reward += ep_reward

        if ep % max(1, n_episodes // 20) == 0 or ep == n_episodes:
            elapsed = time.perf_counter() - t_start
            print(
                f"Ep {ep:4d}/{n_episodes}  "
                f"reward={ep_reward:7.2f}  "
                f"pol_loss={pol_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"entropy={entropy:.3f}  "
                f"elapsed={elapsed:.1f}s"
            )

    print(f"\nTraining complete. Mean reward: {total_reward / n_episodes:.2f}")


# ── Section 8: ONNX export ────────────────────────────────────────────────────

def export_onnx(policy, device, out_path: str) -> None:
    import torch

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dummy = torch.zeros(1, STATE_DIM, dtype=torch.float32, device=device)
    torch.onnx.export(
        policy,
        dummy,
        out_path,
        input_names=["state"],
        output_names=["logits"],
        dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"ONNX model exported → {out_path}  ({size_kb:.1f} KB)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train frontier-selector PPO policy")
    parser.add_argument("--episodes", type=int,   default=500,   help="Training episodes")
    parser.add_argument("--steps",    type=int,   default=256,   help="Steps per episode")
    parser.add_argument("--lr",       type=float, default=LR,    help="Learning rate")
    parser.add_argument("--out",      type=str,   default="models/frontier_selector.onnx",
                        help="Output ONNX path")
    parser.add_argument("--skip-gpu-check", action="store_true",
                        help="Skip GPU verification (use if running CPU-only)")
    parser.add_argument("--no-tb", action="store_true",
                        help="Disable TensorBoard logging")
    parser.add_argument("--run-name", type=str, default="",
                        help="TensorBoard run name suffix (default: timestamp)")
    parser.add_argument("--img-every", type=int, default=10,
                        help="Log map+lidar images to TensorBoard every N episodes (0=off)")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save checkpoint every N episodes (0=off)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint file (.pt)")
    args = parser.parse_args()

    if args.skip_gpu_check:
        import torch
        device = torch.device("cpu")
        print("GPU check skipped — running on CPU")
    else:
        device = check_gpu()

    import torch
    import torch.optim as optim

    policy, value_net = build_networks(device)
    optimiser = optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=args.lr,
    )

    start_ep = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        value_net.load_state_dict(ckpt["value"])
        optimiser.load_state_dict(ckpt["optimiser"])
        start_ep = ckpt["episode"] + 1
        print(f"Resumed from {args.resume} at episode {ckpt['episode']}", flush=True)

    env = FrontierEnv(seed=42)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = None
    if not args.no_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            run_tag = args.run_name or time.strftime("%Y%m%d_%H%M%S")
            log_dir = f"runs/frontier_selector/{run_tag}"
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard: tensorboard --logdir runs/frontier_selector", flush=True)
            print(f"  writing to {log_dir}", flush=True)
            print()
        except ImportError:
            print("TensorBoard not installed — run: pip install tensorboard")
            print("Continuing without visualization.")
            print()

    print(f"Training: {args.episodes} episodes, {args.steps} steps/episode", flush=True)
    print(f"Device:   {device}", flush=True)
    print(f"Params:   policy={sum(p.numel() for p in policy.parameters())}, "
          f"value={sum(p.numel() for p in value_net.parameters())}", flush=True)
    print()

    total_reward = 0.0
    t_start = time.perf_counter()

    os.makedirs("checkpoints", exist_ok=True)

    for ep in range(start_ep, args.episodes + 1):
        rollout = collect_rollout(env, policy, value_net, args.steps, device)

        last_s = torch.tensor(rollout.states[-1], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            last_v = value_net(last_s).item()

        advantages, returns = compute_gae(
            rollout.rewards, rollout.values, rollout.dones, last_v
        )
        pol_loss, val_loss, entropy = ppo_update(
            policy, value_net, optimiser, rollout, advantages, returns, device
        )

        ep_reward = sum(rollout.rewards)
        total_reward += ep_reward
        n_collisions = sum(1 for r in rollout.rewards if r <= -4.9)

        now        = time.time()
        elapsed_s  = time.perf_counter() - t_start
        wall_clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

        if writer is not None:
            writer.add_scalar("reward/episode",       ep_reward,          ep)
            writer.add_scalar("reward/mean_all",      total_reward / ep,  ep)
            writer.add_scalar("loss/policy",          pol_loss,           ep)
            writer.add_scalar("loss/value",           val_loss,           ep)
            writer.add_scalar("policy/entropy",       entropy,            ep)
            writer.add_scalar("env/explored_cells",   env.explored_cells, ep)
            writer.add_scalar("env/collisions",       n_collisions,       ep)
            writer.add_scalar("perf/elapsed_s",       elapsed_s,          ep)
            writer.add_text("info/wall_clock",        wall_clock,         ep)
            if args.img_every > 0 and ep % args.img_every == 0:
                writer.add_image("view/combined", env.render_combined(), ep, dataformats="HWC")
                writer.add_image("view/map",      env.render_map(),      ep, dataformats="HWC")
                writer.add_image("view/lidar",    env.render_lidar(),    ep, dataformats="HWC")
            writer.flush()

        if args.checkpoint_every > 0 and ep % args.checkpoint_every == 0:
            ckpt_path = f"checkpoints/frontier_selector_ep{ep:04d}.pt"
            torch.save({
                "episode":   ep,
                "policy":    policy.state_dict(),
                "value":     value_net.state_dict(),
                "optimiser": optimiser.state_dict(),
            }, ckpt_path)
            print(f"  checkpoint saved → {ckpt_path}", flush=True)

        log_every = max(1, args.episodes // 20)
        if ep % log_every == 0 or ep == args.episodes:
            elapsed = time.perf_counter() - t_start
            print(
                f"Ep {ep:4d}/{args.episodes}  "
                f"reward={ep_reward:7.2f}  "
                f"pol_loss={pol_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"entropy={entropy:.3f}  "
                f"explored={env.explored_cells:5d}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(f"\nTraining complete. Mean reward: {total_reward / args.episodes:.2f}")
    if writer is not None:
        writer.close()

    export_onnx(policy, device, args.out)


if __name__ == "__main__":
    main()
