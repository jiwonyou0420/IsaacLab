# expects SKRL policy(RL, trained with cube position information)(teacher policy)
# dataset generated from rollout of above policy: using franka_cam_record.py
# produces distilled policy that takes camera images as input, doesn't require true position of cube

import argparse, os, csv, math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.models as tvm
SKRL_MODEL_CHECKPOINT = "/home/andres/Documents/jiwon/IsaacLab/logs/skrl/franka_lift/2025-08-11_09-54-01_ppo_torch/checkpoints/best_agent.pt"
DATA_ROOT = "/home/andres/Documents/jiwon/IsaacLab/logs/CREATE/franka_cam_record/2025-08-12_17-35-15"

parser = argparse.ArgumentParser("Distill vision policy from recorded FrankaCam dataset")
parser.add_argument("--teacher", type=str, default = SKRL_MODEL_CHECKPOINT, help="Path to teacher best_agent.pt")
parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Path to recorded run folder (contains dataset.csv)")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--workers", type=int, default=6)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

DATA = Path(args.data_root)
CSV_PATH = DATA / "dataset.csv"

# ----- Dataset -----
class FrankaCamDataset(Dataset):
    def __init__(self, csv_path: Path, root: Path, resize=224, center_crop=False, train=True):
        self.root = root
        rows = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            # find column indices
            sim_t_idx = header.index("simulation_time")
            idx_idx   = header.index("index")
            env_idx   = header.index("env")
            # joint/action columns
            jstart = header.index("panda_joint1")
            jend   = header.index("panda_finger") + 1
            # goal columns
            gx = header.index("goal_x"); gy = header.index("goal_y"); gz = header.index("goal_z")

            for r in reader:
                step = int(r[idx_idx]); env = int(r[env_idx])
                act  = [float(x) for x in r[jstart:jend]]  # 8 dims
                goal = [float(r[gx]), float(r[gy]), float(r[gz])]
                rows.append((step, env, np.array(act, np.float32), np.array(goal, np.float32)))
        self.rows = rows
        # transforms
        base = [T.Resize((resize, resize))]
        aug  = []
        if train:
            aug = [T.ColorJitter(0.1, 0.1, 0.1, 0.05)]
        self.tx = T.Compose(base + aug + [T.ToTensor(),
                        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        step, env, act, goal = self.rows[i]
        env_dir = f"env_{env:03d}"
        name = f"image_{step:06d}.png"
        wrist = Image.open(self.root / "wrist_rgb" / env_dir / name).convert("RGB")
        bird  = Image.open(self.root / "bird_rgb"  / env_dir / name).convert("RGB")
        wrist = self.tx(wrist)  # [3,224,224]
        bird  = self.tx(bird)
        goal  = torch.from_numpy(goal)         # [3]
        act   = torch.from_numpy(act)          # [8]
        return wrist, bird, goal, act

# Simple split (shuffle once)
full = FrankaCamDataset(CSV_PATH, DATA, train=True)
n = len(full)
n_train = int(0.95 * n)
indices = torch.randperm(n).tolist()
train_ds = torch.utils.data.Subset(full, indices[:n_train])
val_ds   = torch.utils.data.Subset(FrankaCamDataset(CSV_PATH, DATA, train=False), indices[n_train:])

train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

# ----- Model -----
class VisionEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, out_dim)
        self.net = m
    def forward(self, x):  # Bx3x224x224
        return self.net(x)

class DistilledPolicy(nn.Module):
    def __init__(self, img_dim=256, cmd_dim=3, proprio_dim=0, hidden=256, out_dim=8):
        super().__init__()
        self.wrist_enc = VisionEncoder(out_dim=img_dim)
        self.bird_enc  = VisionEncoder(out_dim=img_dim)
        self.cmd_proj  = nn.Sequential(nn.Linear(cmd_dim, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
        fuse_in = img_dim*2 + 64 + proprio_dim
        self.head = nn.Sequential(
            nn.Linear(fuse_in, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, wrist, bird, goal, proprio=None):
        wf = self.wrist_enc(wrist)
        bf = self.bird_enc(bird)
        cf = self.cmd_proj(goal)
        if proprio is None:
            x = torch.cat([wf, bf, cf], dim=1)
        else:
            x = torch.cat([wf, bf, cf, proprio], dim=1)
        return self.head(x)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
model = DistilledPolicy().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
mse = nn.MSELoss()

# ----- Train -----
def run_epoch(loader, train=True):
    model.train(train)
    tot, count = 0.0, 0
    for wrist, bird, goal, act in loader:
        wrist = wrist.to(device, non_blocking=True)
        bird  = bird.to(device, non_blocking=True)
        goal  = goal.to(device, non_blocking=True)
        act   = act.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            pred = model(wrist, bird, goal)
            loss = mse(pred, act)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
        bs = act.size(0); tot += loss.item() * bs; count += bs
    return tot / max(1, count)

for ep in range(1, args.epochs + 1):
    tr = run_epoch(train_loader, train=True)
    va = run_epoch(val_loader,   train=False)
    sched.step()
    print(f"[{ep:03d}/{args.epochs}] train {tr:.5f}  val {va:.5f}  lr {sched.get_last_lr()[0]:.2e}")

# ----- Save next to teacher checkpoint -----
ckpt_dir = Path(args.teacher).resolve().parent
save_path = ckpt_dir / "distilled_model.pt"

# save torchscript (easy to deploy)
model.eval()
ex_w = torch.zeros(1, 3, 224, 224, device=device)
ex_b = torch.zeros(1, 3, 224, 224, device=device)
ex_g = torch.zeros(1, 3, device=device)
ts = torch.jit.trace(model, (ex_w, ex_b, ex_g))
ts.save(str(save_path))
print(f"[OK] Saved TorchScript distilled model to: {save_path}")
