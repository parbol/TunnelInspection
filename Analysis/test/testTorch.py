from pathlib import Path
import sys, os

# Get the absolute path of the directory containing this notebook (.../Analysis/test)
project_root = Path.cwd().parent.parent

# Add TunnelInspection to sys.path if it isn't already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Check what path was added (optional verification)
print(f"Added to path: {project_root}")

# Now your imports will work
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DatasetSelector:

    def __init__(self, limits, detsize):
        self.detsize = detsize
        self.xmin = limits[0][0]
        self.xmax = limits[0][1]
        self.ymin = limits[1][0]
        self.ymax = limits[1][1]

    def get(self, data):
        print('xmin:', self.xmin, 'xmax:', self.xmax, 'ymin:', self.ymin, 'ymax:', self.ymax)
        preselect = data[(data[:, 0] > self.xmin) & (data[:, 0] < self.xmax) & (data[:,1] > self.ymin) & (data[:,1] < self.ymax)]
        print(preselect.shape)
        select = preselect[(preselect[:,0] - self.detsize/preselect[:, 5] * preselect[:,3] > self.xmin) &
                           (preselect[:,0] - self.detsize/preselect[:, 5] * preselect[:,3] < self.xmax) &
                           (preselect[:,1] - self.detsize/preselect[:, 5] * preselect[:,4] > self.ymin) &
                           (preselect[:,1] - self.detsize/preselect[:, 5] * preselect[:,4] < self.ymax)]
        return select


def load_to_memmap(file_list, filename='opensky_combined.dat', dtype=np.float32, N=-1):
    # Determine total dimensions
    shapes = []
    for f in file_list:
        with h5py.File(f, 'r') as h5:
            shape = h5['df/block0_values'].shape
            if shape[0] < shape[1]:
                shape = (shape[1], shape[0])
            shapes.append(shape)

    total_rows = sum(s[0] for s in shapes)
    n_features = shapes[0][1]

    print(f"Data to load: {file_list}")
    print(f"  Total number of rows: {total_rows}")
    print(f"  Number of features:   {n_features}")
    if N>0: print(f"   Will load {N} events")

    # Create memory-mapped file on disk
    if N>0 and total_rows>N:
        mmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=(N, n_features))
    else:
        mmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_rows, n_features))

    current_idx = 0
    for f, shape in zip(file_list, shapes):
        with h5py.File(f, 'r') as h5:
            data = h5['df/block0_values'][:]
            if data.shape[0] < data.shape[1]:
                data = data.T
            rows = shape[0]
            if N>0 and current_idx+rows>N:
                mmap_array[current_idx:N] = data[0:N,:].astype(dtype, copy=False)
                current_idx = N
            else:
                mmap_array[current_idx:current_idx + rows] = data.astype(dtype, copy=False)
                current_idx += rows

            
    # Flush changes to disk
    mmap_array.flush()
    return mmap_array


class DifferentiableHistogram2D(nn.Module):

    def __init__(
        self,
        bins: tuple[int, int] = (50, 50),
        x_range: tuple[float, float] = (-1500.0, 1500.0),
        y_range: tuple[float, float] = (-1500.0, 1500.0),
    ):
        super().__init__()
        self.H_bins, self.W_bins = bins
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

    def forward(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Args:

        x_coords: 1D Tensor of shape (N,)
        y_coords: 1D Tensor of shape (N,)
        weights: Optional 1D Tensor of shape (N,)
        Returns:
            2D continuous histogram of shape (H_bins, W_bins)
        """
        if weights is None:
            weights = torch.ones_like(x_coords)

        # 1. Normalize coordinates to continuous grid index coordinates [0, bins-1]
        x_norm = (x_coords - self.x_min) / (self.x_max - self.x_min) * (
            self.W_bins - 1
        )
        y_norm = (y_coords - self.y_min) / (self.y_max - self.y_min) * (
            self.H_bins - 1
        )

        # 2. Get bounding 4 neighbor cell indices
        x0 = torch.floor(x_norm).long()
        y0 = torch.floor(y_norm).long()
        x1 = x0 + 1
        y1 = y0 + 1

        # Clip indices to grid boundaries to avoid indexing errors
        x0_c = torch.clamp(x0, 0, self.W_bins - 1)
        x1_c = torch.clamp(x1, 0, self.W_bins - 1)
        y0_c = torch.clamp(y0, 0, self.H_bins - 1)
        y1_c = torch.clamp(y1, 0, self.H_bins - 1)

        # 3. Bilinear interpolation weights (differentiable w.r.t coordinates)
        w_x1 = x_norm - x0.float()
        w_x0 = 1.0 - w_x1
        w_y1 = y_norm - y0.float()
        w_y0 = 1.0 - w_y1

        w00 = weights * w_x0 * w_y0
        w01 = weights * w_x0 * w_y1
        w10 = weights * w_x1 * w_y0
        w11 = weights * w_x1 * w_y1

        # 4. Scatter add into 2D grid
        hist = torch.zeros(
            (self.H_bins, self.W_bins),
            dtype=x_coords.dtype,
            device=x_coords.device,
        )

        hist.index_put_((y0_c, x0_c), w00, accumulate=True)
        hist.index_put_((y1_c, x0_c), w01, accumulate=True)
        hist.index_put_((y0_c, x1_c), w10, accumulate=True)
        hist.index_put_((y1_c, x1_c), w11, accumulate=True)

        return hist


class ChamberBlindDetectorLoss(nn.Module):

    def __init__(self, kernel_size: int = 15, sigma: float = 3.0, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

        # Create a 2D Gaussian smoothing kernel
        ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()

        # Shape (out_channels, in_channels, H, W) for conv2d
        self.register_buffer(
            "kernel", kernel.view(1, 1, kernel_size, kernel_size)
        )

    def forward(self, ratio_map: torch.Tensor, valid_mask: torch.Tensor = None):
        """Args:

        ratio_map: Tensor of shape (1, 1, H, W) or (H, W)
        valid_mask: Optional boolean mask (1 for valid sky bins, 0 for edge
        voids)
        """
        if ratio_map.ndim == 2:
            ratio_map = ratio_map.unsqueeze(0).unsqueeze(0)

        # 1. Spatially smooth to aggregate chamber signal and suppress single-pixel noise
        smoothed = F.conv2d(ratio_map, self.kernel, padding="same")

        if valid_mask is not None:
            if valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
            smoothed = smoothed * valid_mask
            flat_smoothed = smoothed[valid_mask > 0]
        else:
            flat_smoothed = smoothed.flatten()

        # 2. Background statistics
        mu_bg = torch.mean(flat_smoothed)
        sigma_bg = torch.std(flat_smoothed) + 1e-6

        # 3. SoftMax-based peak approximation (smooth, differentiable max)
        # LogSumExp gives a smooth approximation of max(smoothed)
        peak_val = (
            self.temperature
            * torch.logsumexp(flat_smoothed / self.temperature, dim=0)
            - self.temperature * torch.log(torch.tensor(flat_smoothed.numel()))
        )

        # 4. Significance Z-score
        z_score = (peak_val - mu_bg) / sigma_bg

        # Return negative Z-score for minimization
        return -z_score


def project_muons(data, detPos, mode='sky', detSize=100., detSizeZ=40., Z=1000.):
    # Compute limits of detector
    limits = []
    for i,pos in enumerate(detPos):
        lim_temp = []
        lim_temp.append((pos[0]-detSize/2.,pos[0]+detSize/2.))
        lim_temp.append((pos[1]-detSize/2.,pos[1]+detSize/2.))
        limits.append(lim_temp)

    dataMeas = []
    for i,lim in enumerate(limits):
        dsel_temp = DatasetSelector(limits=(lim[0],lim[1]), detsize=detSizeZ)
        dataMeas.append(dsel_temp.get(data))

    # Get 2D histograms
    totalDataX, totalDataY = [], []
    for i in range(len(dataMeas)):
        data_temp = dataMeas[i]

        l = (Z-data_temp[:,2])/data_temp[:,5]
        x = data_temp[:,0] + l * data_temp[:,3]
        y = data_temp[:,1] + l * data_temp[:,4]

        totalDataX += list(x)
        totalDataY += list(y)

    totalDataX = torch.tensor(totalDataX)
    totalDataY = torch.tensor(totalDataY)
    return totalDataX, totalDataY


def soft_box_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    center: torch.Tensor,
    size_xy: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Computes a smooth differentiable acceptance weight in [0, 1]

    for points (x, y) falling inside the detector box centered at `center`.
    """
    half_w = size_xy / 2.0
    x_min, x_max = center[0] - half_w, center[0] + half_w
    y_min, y_max = center[1] - half_w, center[1] + half_w

    # Smooth step function using sigmoids
    in_x = torch.sigmoid((x - x_min) / temperature) * torch.sigmoid(
        (x_max - x) / temperature
    )
    in_y = torch.sigmoid((y - y_min) / temperature) * torch.sigmoid(
        (y_max - y) / temperature
    )

    return in_x * in_y


def project_muons_differentiable(
    data: torch.Tensor,
    detPos: torch.Tensor,
    detSize: float = 100.0,
    detSizeZ: float = 40.0,
    Z: float = 1000.0,
    temp: float = 2.0,
):
    """Computes projection coordinates and continuous differentiable detector acceptance weights.

    data columns: [x0, y0, z0, dx, dy, dz]
    """
    # 1. Project all tracks to the target Z plane
    dz = data[:, 5]
    l = (Z - data[:, 2]) / dz
    x_proj = data[:, 0] + l * data[:, 3]
    y_proj = data[:, 1] + l * data[:, 4]

    # Detector acceptance is evaluated at top and bottom detector layers
    # Layer 1 (z = z0):
    x_top = data[:, 0]
    y_top = data[:, 1]
    # Layer 2 (z = z0 - detSizeZ):
    x_bot = data[:, 0] - (detSizeZ / dz) * data[:, 3]
    y_bot = data[:, 1] - (detSizeZ / dz) * data[:, 4]

    # 2. Accumulate differentiable weights across all 3 detector positions
    total_weights = torch.zeros(
        data.shape[0], device=data.device, dtype=data.dtype
    )

    for k in range(detPos.shape[0]):
        pos = detPos[k]
        w_top = soft_box_2d(x_top, y_top, pos, detSize, temperature=temp)
        w_bot = soft_box_2d(x_bot, y_bot, pos, detSize, temperature=temp)
        # Muon must hit both upper and lower panels
        total_weights = total_weights + (w_top * w_bot)

    return x_proj, y_proj, total_weights


def prefilter_dataset(
    data: torch.Tensor,
    det_search_bbox: tuple[float, float, float, float],
    margin: float = 300.0,
) -> torch.Tensor:
    """Pre-filters the static dataset ONCE using a broad bounding box.

    Muons outside this wide bounding box can never hit the detectors during
    optimization, so we discard them to save RAM without affecting gradients.
    det_search_bbox: (xmin, xmax, ymin, ymax) of the allowable search area.
    """
    xmin, xmax, ymin, ymax = det_search_bbox
    mask = (
        (data[:, 0] >= xmin - margin)
        & (data[:, 0] <= xmax + margin)
        & (data[:, 1] >= ymin - margin)
        & (data[:, 1] <= ymax + margin)
    )
    return data[mask].clone()  # Clone frees up unreferenced slices


def project_muons_chunked(
    data: torch.Tensor,
    detPos: torch.Tensor,
    hist_builder: nn.Module,
    detSize: float = 100.0,
    detSizeZ: float = 40.0,
    Z: float = 1000.0,
    temp: float = 2.0,
    chunk_size: int = 100_000,
) -> torch.Tensor:
    """Computes differentiable projection and directly accumulates into the 2D histogram

    in small chunks to keep memory usage under a few hundred megabytes.
    """
    total_rows = data.shape[0]
    hist_acc = torch.zeros(
        (hist_builder.H_bins, hist_builder.W_bins),
        dtype=data.dtype,
        device=detPos.device,
    )

    # Process in memory-friendly chunks
    for i in range(0, total_rows, chunk_size):
        batch = data[i : i + chunk_size]

        dz = batch[:, 5]
        l = (Z - batch[:, 2]) / dz
        x_proj = batch[:, 0] + l * batch[:, 3]
        y_proj = batch[:, 1] + l * batch[:, 4]

        x_top, y_top = batch[:, 0], batch[:, 1]
        x_bot = batch[:, 0] - (detSizeZ / dz) * batch[:, 3]
        y_bot = batch[:, 1] - (detSizeZ / dz) * batch[:, 4]

        weights = torch.zeros(batch.shape[0], dtype=data.dtype, device=data.device)

        for k in range(detPos.shape[0]):
            pos = detPos[k]
            w_top = soft_box_2d(x_top, y_top, pos, detSize, temperature=temp)
            w_bot = soft_box_2d(x_bot, y_bot, pos, detSize, temperature=temp)
            weights = weights + (w_top * w_bot)

        # Splat directly into histogram to discard per-event tensors immediately
        hist_batch = hist_builder(x_proj, y_proj, weights=weights)
        hist_acc = hist_acc + hist_batch

    return hist_acc


if __name__ == '__main__':
    print(">>>> Loading data into memmap...")
    NEVENTS = 30000000
    openskyFiles = ['/home/ruben/Documents/TunnelInspection/data/outputOpensky_v3_249p8_compressed.h5']
    tunnelFiles  = ['/home/ruben/Documents/TunnelInspection/data/outputTunnel_v4_596p6_compressed.h5']

    openskyfull = load_to_memmap(openskyFiles, 'opensky.dat', N=NEVENTS)
    tunnelfull  = load_to_memmap(tunnelFiles, 'tunnel.dat', N=NEVENTS)

    # 1. Pre-filter broadly once on disk/CPU before converting to PyTorch tensors
    tunnel_bounds = [-400.0, 400.0, -1000.0, 1000.0]
    print(">>>> Pre-filtering dataset around region of interest...")
    openskyTensor = prefilter_dataset(torch.from_numpy(openskyfull), tunnel_bounds, margin=300.0)
    tunnelTensor  = prefilter_dataset(torch.from_numpy(tunnelfull),  tunnel_bounds, margin=300.0)
    print(f"Filtered Tunnel shape: {tunnelTensor.shape}, OpenSky shape: {openskyTensor.shape}")

    # Move to GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    tunnelTensor = tunnelTensor.to(device)
    openskyTensor = openskyTensor.to(device)

    # 2. Setup parameters and optimizer
    initDetectorPos = [[-100.0, -100.0], [-100.0, 100.0], [100.0, 100.0]]
    detector_pos = torch.nn.Parameter(
        torch.tensor(initDetectorPos, dtype=torch.float32, device=device, requires_grad=True)
    )
    optimizer = torch.optim.Adam([detector_pos], lr=4.0)

    hist_builder = DifferentiableHistogram2D(
        bins=(50, 50), x_range=(-1500.0, 1500.0), y_range=(-1500.0, 1500.0)
    ).to(device)
    blind_loss = ChamberBlindDetectorLoss(kernel_size=15, sigma=3.0).to(device)

    # 3. Iterative Loop
    for epoch in range(50):
        optimizer.zero_grad()

        # Differentiable forward pass accumulated in small chunks
        H_tunnel = project_muons_chunked(tunnelTensor, detector_pos, hist_builder, chunk_size=100000)
        H_sky    = project_muons_chunked(openskyTensor, detector_pos, hist_builder, chunk_size=100000)

        ratio_map = H_tunnel / (H_sky + 1e-3)
        #valid_mask = (ratio_map > 0.9*torch.max(ratio_map)).float()
        #print(valid_mask)
        #print(H_tunnel, torch.max(H_tunnel))
        #print(H_sky, torch.max(H_sky))

        # Plot epoch ratio hist
        fig,ax = plt.subplots(2, 1, figsize = (8, 12))
        xedges = np.linspace(-1500.0, 1500.0, 51)
        yedges = np.linspace(-1500.0, 1500.0, 51)
        pc = ax[0].pcolorfast(xedges, yedges, ratio_map.detach().cpu().numpy().T)
        fig.colorbar(pc, ax=ax[0])
        smoothed = F.conv2d(ratio_map.unsqueeze(0).unsqueeze(0), blind_loss.kernel, padding="same")
        pc1 = ax[1].pcolorfast(xedges, yedges, np.array(smoothed[0][0].detach().cpu().numpy()).T)
        fig.colorbar(pc1, ax=ax[1])
        plt.savefig(f'plots/epoch_{epoch}.png')
        plt.close()

        loss = blind_loss(ratio_map)
        loss.backward()
        optimizer.step()

        # Enforce search boundaries
        with torch.no_grad():
            detector_pos[:, 0].clamp_(min=tunnel_bounds[0]+50., max=tunnel_bounds[1]-50.)
            detector_pos[:, 1].clamp_(min=tunnel_bounds[2]+50., max=tunnel_bounds[3]-50.)

        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")
        print(f"                      Detector 0 position: {detector_pos[0].detach().cpu().numpy()}")
        print(f"                      Detector 1 position: {detector_pos[1].detach().cpu().numpy()}")
        print(f"                      Detector 2 position: {detector_pos[2].detach().cpu().numpy()}")
