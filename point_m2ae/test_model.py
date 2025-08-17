# point_m2ae/test_model.py
# -*- coding: utf-8 -*-
"""
Quick encoder-only feature extraction with Point-M2AE on sparse voxel grids.

Expected layout:
  ~/adaptfoundation_linearprobing/
    ├── crops/2mm/S.Or./ ...
    └── point_m2ae/
        ├── test_model.py        <-- this file
        ├── Point-M2AE/          <-- upstream repo
        └── ckpt/pre-train.pth   <-- pretrained weights

Run from the project root:
  $ cd ~/adaptfoundation_linearprobing
  $ python -u point_m2ae/test_model.py

Notes:
- Group() in this repo is KNN/FPS-based and does NOT take a radius argument.
- Defaults are conservative for ~600–800 active voxels.
- We return four features per sample: mean, max, (mean+max), and concat(mean,max).
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

# ---------- Paths & imports setup ----------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]                     # ~/adaptfoundation_linearprobing
PM2AE_DIR = HERE.parent / "Point-M2AE"     # point_m2ae/Point-M2AE

for p in (ROOT, PM2AE_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

print(f"📄 test_model.py path check: {HERE} exists? {HERE.exists()}")
print(f"📄 test_model.py path check: {HERE} exists? {HERE.exists()}")
print(f"   • ROOT     = {ROOT}")
print(f"   • PM2AE_DIR= {PM2AE_DIR}")

# --- KNN compatibility shim (put this BEFORE 'from models.modules import Group') ---
import importlib, inspect
mods = importlib.import_module('models.modules')  # module qui contient Group et utilise 'knn'

try:
    from knn_cuda import knn as _knn_native
    sig = inspect.signature(_knn_native)
    if len(sig.parameters) == 3:
        # Le repo appelle knn(x, y, k, transpose_mode=True).
        # Avec knn_cuda==0.2 installée, l'API peut n'exposer que 3 args → compat.
        def _knn_compat(x, y, k, transpose_mode=True):
            return _knn_native(x, y, k)
        mods.knn = _knn_compat
        print("🔧 Applied knn() compatibility shim (4→3 args).")
except Exception as e:
    print(f"⚠️ Could not apply KNN shim: {e}")
# -------------------------------------------------------------------------------

# Point-M2AE modules
from models.Point_M2AE_Finetune import H_Encoder
from models.modules import Group

# Your dataloader
from data.loaders import HCPOFCDataLoader


# ---------- Simple config object ----------
class Cfg:
    # Encoder hierarchy (repo defaults)
    encoder_depths = [5, 5, 5]
    encoder_dims   = [96, 192, 384]

    # Grouping (KNN via Group; no radius here)
    group_sizes = [16, 16, 16]
    num_groups  = [256, 128, 32]

    # Local radii consumed by H_Encoder internals (keep repo values)
    local_radius = [0.32, 0.64, 1.28]

    # Misc
    drop_path_rate = 0.1
    num_heads      = 6


# ---------- Utils ----------
def torch_info():
    print("—— Torch/CUDA ——————————————————")
    print(f"• torch.__version__      : {torch.__version__}")
    print(f"• torch.version.cuda     : {torch.version.cuda}")
    print(f"• torch.cuda.is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"• torch.cuda.get_device  : {torch.cuda.get_device_name(0)}")
    print("———————————————————————————")

def preprocess_roi_to_points(roi_3d: np.ndarray) -> np.ndarray:
    """
    Binary ROI (Z,Y,X) -> (N,3) float32 points scaled to ~[-1,1].
    """
    if roi_3d.ndim != 3:
        raise ValueError(f"ROI must be 3D (got shape {roi_3d.shape}).")
    idx = np.argwhere(roi_3d == 1)  # (N, 3), int
    if idx.size == 0:
        raise ValueError("ROI has 0 active voxels.")
    pts = idx.astype(np.float32)
    pts -= pts.mean(axis=0, keepdims=True)
    scale = float(np.abs(pts).max()) + 1e-6
    pts /= scale
    return pts  # (N,3)

def make_groupers(cfg: Cfg) -> torch.nn.ModuleList:
    """
    Build Group modules: KNN/FPS grouping (no radius arg in this repo).
    """
    gs = torch.nn.ModuleList()
    for ng, gs_ in zip(cfg.num_groups, cfg.group_sizes):
        gs.append(Group(num_group=ng, group_size=gs_))
    return gs


# ---------- Debug helpers ----------
def _check_finite(t: torch.Tensor, name: str) -> bool:
    ok = torch.isfinite(t).all().item()
    if not ok:
        nbad = (~torch.isfinite(t)).sum().item()
        print(f"⚠️  {name}: {nbad}/{t.numel()} éléments non finis (NaN/Inf).")
    return bool(ok)

@torch.no_grad()
def _groups_debug(nei: torch.Tensor, ctr: torch.Tensor, scale: int, eps_var: float = 1e-10):
    """
    nei: (B, G, K, 3), ctr: (B, G, 3)
    Reports:
      - % of groups with near-zero intra-group variance
      - % of groups with exact duplicates among neighbors
      - min / median of minimal pairwise distances per group
    """
    B, G, K, _ = nei.shape
    _check_finite(nei, f"Scale {scale} neighborhoods")
    _check_finite(ctr, f"Scale {scale} centers")

    # Variance intra-groupe
    dif = nei - nei.mean(dim=2, keepdim=True)            # (B,G,K,3)
    var = (dif * dif).mean(dim=(2,3))                    # (B,G)
    near_const = (var < eps_var)                         # (B,G)

    # Distances intra-groupe → doublons
    X = nei.reshape(B * G, K, 3).contiguous()
    d = torch.cdist(X, X)                                # (B*G,K,K)
    d = d + torch.eye(K, device=d.device)[None] * 1e6    # masque diag
    min_d = d.min(dim=-1).values.min(dim=-1).values      # (B*G,)
    min_d = min_d.reshape(B, G)

    dup = (min_d < 1e-8)
    frac_const = near_const.float().mean().item()
    frac_dup   = dup.float().mean().item()
    md_min     = float(min_d.min().item())
    md_med     = float(min_d.median().item())

    print(f"  • Scale {scale} debug:"
          f" near-const={frac_const*100:5.1f}%,"
          f" dup={frac_dup*100:5.1f}%,"
          f" min(min_d)={md_min:.2e},"
          f" median(min_d)={md_med:.2e}")


# ---------- Encoder-only wrapper ----------
class PointM2AEFeatureExtractor(torch.nn.Module):
    def __init__(self, checkpoint_path: Path, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = H_Encoder(cfg).to(self.device).eval()
        self.groupers = make_groupers(cfg).to(self.device).eval()

        self._load_encoder_weights(checkpoint_path)
        print("✅ Encoder ready.")

    def _load_encoder_weights(self, ckpt_path: Path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"• Loading checkpoint: {ckpt_path}")
        obj = torch.load(str(ckpt_path), map_location="cpu")

        # Try typical containers
        sd = None
        for k in ("state_dict", "base_model", "model", "module"):
            if isinstance(obj, dict) and isinstance(obj.get(k, None), dict):
                sd = obj[k]
                break
        if sd is None and isinstance(obj, dict):
            sd = obj

        # Keep only encoder keys, strip 'h_encoder.' prefix
        enc_sd = {}
        for k, v in sd.items():
            if k.startswith("h_encoder."):
                enc_sd[k.replace("h_encoder.", "")] = v

        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        print(f"• load_state_dict(strict=False) → missing={len(missing)}, unexpected={len(unexpected)}")

    @torch.no_grad()
    def forward(self, pts: torch.Tensor):
        """
        pts: (B, N, 3) float32 CUDA
        returns:
          feat_sum  : (B, C)  # mean + max (element-wise)
          feat_mean : (B, C)
          feat_max  : (B, C)
          feat_cat  : (B, 2C) # concat [mean, max]
        """
        assert pts.ndim == 3 and pts.shape[-1] == 3, f"pts shape {pts.shape}"
        pts = pts.contiguous()  # IMPORTANT for pointnet2_ops

        neighborhoods, centers, idxs = [], [], []
        cur = pts
        for i, g in enumerate(self.groupers):
            cur = cur.contiguous()             # ensure contiguity at every hop
            nei, ctr, idx = g(cur)            # (B,G,K,3), (B,G,3)
            neighborhoods.append(nei)
            centers.append(ctr)
            idxs.append(idx)
            cur = ctr

        # Summary + stabilité
        print("— Grouping summary —")
        for i, (nei, ctr) in enumerate(zip(neighborhoods, centers)):
            B, G, K, _ = nei.shape
            print(f"  • Scale {i}: groups={G}, group_size={K}, centers_contiguous={ctr.is_contiguous()}")
            _groups_debug(nei, ctr, i)

        # Encoder → tokens (B, G_last, C)
        x_vis = self.encoder(neighborhoods, centers, idxs, eval=True)
        _check_finite(x_vis, "x_vis (encoder output)")

        feat_mean = x_vis.mean(1)                   # (B, C)
        feat_max  = x_vis.max(1).values             # (B, C)
        feat_sum  = feat_mean + feat_max            # (B, C)
        feat_cat  = torch.cat([feat_mean, feat_max], dim=1)  # (B, 2C)

        for n, t in [("feat_mean", feat_mean), ("feat_max", feat_max),
                     ("feat_sum", feat_sum), ("feat_cat", feat_cat)]:
            _check_finite(t, n)

        return feat_sum, feat_mean, feat_max, feat_cat


# ---------- Main ----------
def main():
    torch_info()

    # Data & checkpoint
    data_dir = ROOT / "crops" / "2mm" / "S.Or."
    ckpt_path = HERE.parent / "ckpt" / "pre-train.pth"
    print(f"• Data dir : {data_dir}")
    print(f"• Checkpoint: {ckpt_path}")

    # Build extractor
    cfg = Cfg()
    extractor = PointM2AEFeatureExtractor(ckpt_path, cfg)

    # Gather 5 samples: real if available, else synthetic fallback
    samples: list[tuple[np.ndarray, int, str]] = []
    use_synth = False
    if data_dir.exists():
        try:
            loader = HCPOFCDataLoader(data_dir)
            tensor_data, labels, subj_ids = loader.load_split_as_tensor("train_val_split_0.csv")
            # tensor_data: (N, 1, 30, 38, 22)
            n_pick = min(5, int(tensor_data.shape[0]))
            for i in range(n_pick):
                vol = tensor_data[i].numpy()  # -> (30,38,22)
                samples.append((vol, int(labels[i]), str(subj_ids[i])))
            print(f"✅ Loaded real samples: {len(samples)}")
        except Exception as e:
            print(f"⚠️  Failed to load dataset split: {e}\n   → Using synthetic fallback.")
            use_synth = True
    else:
        print("⚠️  Data dir not found → Using synthetic fallback.")
        use_synth = True

    if use_synth:
        rng = np.random.default_rng(0)
        for i in range(5):
            vol = np.zeros((30, 38, 22), dtype=np.uint8)
            n = 600
            zz = rng.integers(0, 30, size=n)
            yy = rng.integers(0, 38, size=n)
            xx = rng.integers(0, 22, size=n)
            vol[zz, yy, xx] = 1
            samples.append((vol, -1, f"synth_{i}"))
        print(f"✅ Built synthetic samples: {len(samples)}")

    # Run a few
    all_sum, all_mean, all_max, all_cat = [], [], [], []

    print("\n📊 Extracting features:")
    for i, (roi, label, sid) in enumerate(samples, 1):
        try:
            pts_np = preprocess_roi_to_points(roi)                # (N,3), float32
            pts = torch.from_numpy(pts_np.copy()).to(torch.float32) \
                     .unsqueeze(0).to("cuda").contiguous()        # (1,N,3)

            feat_sum, feat_mean, feat_max, feat_cat = extractor(pts)  # each (1,C) or (1,2C)
            fsum  = feat_sum.squeeze(0).cpu().numpy()             # (C,)
            fmean = feat_mean.squeeze(0).cpu().numpy()            # (C,)
            fmax  = feat_max.squeeze(0).cpu().numpy()             # (C,)
            fcat  = feat_cat.squeeze(0).cpu().numpy()             # (2C,)

            all_sum.append(fsum)
            all_mean.append(fmean)
            all_max.append(fmax)
            all_cat.append(fcat)

            print(f"  • Sample {i}: subject={sid:>10} | label={label:>2} | "
                  f"points={pts.shape[1]:>4} | "
                  f"feat_mean={fmean.shape}, feat_max={fmax.shape}, "
                  f"feat_sum={fsum.shape}, feat_cat={fcat.shape}")
        except Exception as e:
            print(f"  ❌ Sample {i} failed: {e}")

    # Optional: petite agrégation sur les 5 échantillons (moyenne par feature)
    if all_sum:
        A_sum  = np.stack(all_sum,  axis=0)    # (S,C)
        A_mean = np.stack(all_mean, axis=0)    # (S,C)
        A_max  = np.stack(all_max,  axis=0)    # (S,C)
        A_cat  = np.stack(all_cat,  axis=0)    # (S,2C)

        # Sanity NaN/Inf
        for name, A in [("A_sum", A_sum), ("A_mean", A_mean), ("A_max", A_max), ("A_cat", A_cat)]:
            ok = np.isfinite(A).all()
            print(f"  · {name} finite = {ok}")

        print("\n📦 Aggregation over samples (simple mean):")
        print(f"  · mean(A_sum)  shape: {A_sum.mean(0).shape}")
        print(f"  · mean(A_mean) shape: {A_mean.mean(0).shape}")
        print(f"  · mean(A_max)  shape: {A_max.mean(0).shape}")
        print(f"  · mean(A_cat)  shape: {A_cat.mean(0).shape}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# (Tips if you see NaNs/unstable groups)
# - Reduce token pressure for small clouds (~600–700 pts):
#     Cfg.num_groups = [256, 128, 32]
#     Cfg.group_sizes = [16, 16, 16]
# - If still unstable, try even fewer groups (e.g. [128, 64, 16]) or lower K.
# - Keep tensors .contiguous(), eval() mode, and no AMP for this encoder-only path.
# -----------------------------------------------------------------------------
