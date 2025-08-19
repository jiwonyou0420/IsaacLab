# Distill a vision policy from recorded rollouts (robot-agnostic CSV schema)

import argparse, os, csv, re
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.models as tvm

SKRL_MODEL_CHECKPOINT = "/home/andres/Documents/jiwon/IsaacLab/logs/skrl/franka_lift/2025-08-11_09-54-01_ppo_torch/checkpoints/best_agent.pt"
DATA_ROOT = "/home/andres/Documents/jiwon/IsaacLab/logs/CREATE/franka_cam_record/2025-08-19_12-39-57"

parser = argparse.ArgumentParser("Distill vision policy from recorded dataset")
parser.add_argument("--teacher", type=str, default=SKRL_MODEL_CHECKPOINT, help="Path to teacher best_agent.pt")
parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Path to recorded run folder (contains dataset.csv)")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--workers", type=int, default=6)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--resize", type=int, default=224)
parser.add_argument("--use_proprio", default=True, help="Feed joint pos/vel to the policy")
args = parser.parse_args()

DATA = Path(args.data_root)
CSV_PATH = DATA / "dataset.csv"

# ---------- Helpers to parse dynamic headers ----------
_re_joint = re.compile(r"^joint(\d+)_(pos|vel)$")
_re_action = re.compile(r"^action(\d+)$")

def _parse_header(header):
    # mandatory fields
    sim_t_idx = header.index("simulation_time")
    idx_idx   = header.index("index")
    env_idx   = header.index("env")
    gx = header.index("goal_x")
    gy = header.index("goal_y")
    gz = header.index("goal_z")

    joint_pos_cols = []
    joint_vel_cols = []
    action_cols = []

    for i, name in enumerate(header):
        m = _re_joint.match(name)
        if m:
            j = int(m.group(1))
            kind = m.group(2)
            if kind == "pos":
                joint_pos_cols.append((j, i))
            else:
                joint_vel_cols.append((j, i))
            continue
        m2 = _re_action.match(name)
        if m2:
            a = int(m2.group(1))
            action_cols.append((a, i))

    # sort numerically
    joint_pos_cols.sort(key=lambda x: x[0])
    joint_vel_cols.sort(key=lambda x: x[0])
    action_cols.sort(key=lambda x: x[0])

    # return column indices
    pos_idx = [i for _, i in joint_pos_cols]
    vel_idx = [i for _, i in joint_vel_cols]
    act_idx = [i for _, i in action_cols]

    return {
        "sim_time": sim_t_idx,
        "index": idx_idx,
        "env": env_idx,
        "goal": (gx, gy, gz),
        "jpos": pos_idx,
        "jvel": vel_idx,
        "acts": act_idx,
        "n_joints": max(len(pos_idx), len(vel_idx)),
        "n_actions": len(act_idx),
    }

# ---------- Dataset ----------
class VisionDistillDataset(Dataset):
    def __init__(self, csv_path: Path, root: Path, resize=224, train=True, use_proprio=False):
        self.root = root
        self.use_proprio = use_proprio
        rows = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            idxs = _parse_header(header)

            for r in reader:
                step = int(r[idxs["index"]])
                env  = int(r[idxs["env"]])

                # goal (3,)
                gx, gy, gz = idxs["goal"]
                goal = np.array([float(r[gx]), float(r[gy]), float(r[gz])], dtype=np.float32)

                # actions (A,)
                actions = np.array([float(r[i]) for i in idxs["acts"]], dtype=np.float32)

                # optional proprio (2*N,)
                proprio = None
                if self.use_proprio:
                    jpos = np.array([float(r[i]) for i in idxs["jpos"]], dtype=np.float32)
                    jvel = np.array([float(r[i]) for i in idxs["jvel"]], dtype=np.float32)
                    proprio = np.concatenate([jpos, jvel], axis=0).astype(np.float32)

                rows.append((step, env, goal, actions, proprio))

        self.rows = rows
        self.meta = {
            "n_actions": len(_parse_header(open(csv_path).readline().strip().split(","))["acts"])
        }  # not used later; act dim is inferred during model ctor below

        # transforms
        txs = [T.Resize((resize, resize))]
        if train:
            txs += [T.ColorJitter(0.1, 0.1, 0.1, 0.05)]
        txs += [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])]
        self.tx = T.Compose(txs)
        print("Dataset init complete")

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        step, env, goal, actions, proprio = self.rows[i]
        env_dir = f"env_{env:03d}"
        name = f"image_{step:06d}.png"

        wrist = Image.open(self.root / "wrist_rgb" / env_dir / name).convert("RGB")
        bird  = Image.open(self.root / "bird_rgb"  / env_dir / name).convert("RGB")

        wrist = self.tx(wrist)  # [3,H,W]
        bird  = self.tx(bird)

        goal_t = torch.from_numpy(goal)              # [3]
        act_t  = torch.from_numpy(actions)           # [A]
        if proprio is None:
            return wrist, bird, goal_t, act_t, None
        else:
            proprio_t = torch.from_numpy(proprio)    # [2*N]
            return wrist, bird, goal_t, act_t, proprio_t

# ---------- Build datasets ----------
full = VisionDistillDataset(CSV_PATH, DATA, resize=args.resize, train=True, use_proprio=args.use_proprio)
n = len(full)
n_train = int(0.95 * n)
indices = torch.randperm(n).tolist()
train_ds = torch.utils.data.Subset(full, indices[:n_train])

# For val we re-open with train=False to turn off color jitter
val_full = VisionDistillDataset(CSV_PATH, DATA, resize=args.resize, train=False, use_proprio=args.use_proprio)
val_ds   = torch.utils.data.Subset(val_full, indices[n_train:])

train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=args.workers, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
print("Data loaders complete")
# Infer action dim (robustly grab first batch)
def peek_action_dim(loader):
    for _, _, _, act, _ in loader:
        return act.shape[1]
    raise RuntimeError("Empty dataset")

act_dim = peek_action_dim(train_loader)

# If using proprio, infer its dim
def peek_proprio_dim(loader):
    for *_rest, proprio in loader:
        if proprio is None:
            return 0
        return proprio.shape[1]
    return 0

proprio_dim = peek_proprio_dim(train_loader) if args.use_proprio else 0

# ---------- Model ----------
class VisionEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, out_dim)
        self.net = m
    def forward(self, x):  # Bx3xHxW
        return self.net(x)

class DistilledPolicy(nn.Module):
    def __init__(self, img_dim=256, cmd_dim=3, proprio_dim=0, out_dim=8):
        super().__init__()
        self.wrist_enc = VisionEncoder(out_dim=img_dim)
        self.bird_enc  = VisionEncoder(out_dim=img_dim)
        self.cmd_proj  = nn.Sequential(
            nn.Linear(cmd_dim, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU()
        )
        fuse_in = img_dim * 2 + 64 + proprio_dim
        self.head = nn.Sequential(
            nn.Linear(fuse_in, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, out_dim),
        )
        self._proprio_dim = proprio_dim

    def forward(self, wrist, bird, goal, proprio=None):
        wf = self.wrist_enc(wrist)
        bf = self.bird_enc(bird)
        cf = self.cmd_proj(goal)
        if self._proprio_dim > 0 and proprio is not None:
            x = torch.cat([wf, bf, cf, proprio], dim=1)
        else:
            x = torch.cat([wf, bf, cf], dim=1)
        return self.head(x)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
model = DistilledPolicy(proprio_dim=proprio_dim, out_dim=act_dim).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
mse = nn.MSELoss()

# ---------- Train ----------
def run_epoch(loader, train=True):
    model.train(train)
    tot, count = 0.0, 0
    for wrist, bird, goal, act, proprio in loader:
        wrist = wrist.to(device, non_blocking=True)
        bird  = bird.to(device, non_blocking=True)
        goal  = goal.to(device, non_blocking=True)
        act   = act.to(device, non_blocking=True)
        proprio = None if proprio is None else proprio.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred = model(wrist, bird, goal, proprio)
            loss = mse(pred, act)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
        bs = act.size(0)
        tot += loss.item() * bs
        count += bs
    return tot / max(1, count)

print("Training start")
for ep in range(1, args.epochs + 1):
    tr = run_epoch(train_loader, train=True)
    va = run_epoch(val_loader,   train=False)
    sched.step()
    print(f"[{ep:03d}/{args.epochs}] train {tr:.5f}  val {va:.5f}  lr {sched.get_last_lr()[0]:.2e}")

# ---------- Save next to teacher checkpoint ----------
ckpt_dir = Path(args.teacher).resolve().parent
save_path = ckpt_dir / "distilled_model.pt"

model.eval()
ex_w = torch.zeros(1, 3, args.resize, args.resize, device=device)
ex_b = torch.zeros(1, 3, args.resize, args.resize, device=device)
ex_g = torch.zeros(1, 3, device=device)
if proprio_dim > 0:
    ex_p = torch.zeros(1, proprio_dim, device=device)
    ts = torch.jit.trace(model, (ex_w, ex_b, ex_g, ex_p))
else:
    ts = torch.jit.trace(model, (ex_w, ex_b, ex_g))
ts.save(str(save_path))
print(f"[OK] Saved TorchScript distilled model to: {save_path} (act_dim={act_dim}, proprio_dim={proprio_dim})")
