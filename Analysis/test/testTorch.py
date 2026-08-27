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
import matplotlib.patches as patches
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


class ChamberDoGLoss(nn.Module):

    def __init__(
        self,
        kernel_size: int = 20,
        sigma1: float = 2.0,
        sigma2: float = 5.0,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.temperature = temperature

        # Create coordinate grid
        ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        r2 = xx**2 + yy**2

        # Compute both Gaussian components
        g1 = torch.exp(-r2 / (2.0 * sigma1**2))
        g1 = g1 / g1.sum()

        g2 = torch.exp(-r2 / (2.0 * sigma2**2))
        g2 = g2 / g2.sum()

        # DoG kernel (zero-mean bandpass filter)
        dog_kernel = g1 - g2
        self.register_buffer(
            "kernel", dog_kernel.view(1, 1, kernel_size, kernel_size)
        )

    def forward(
        self,
        ratio_map: torch.Tensor,
        sky_counts: torch.Tensor = None,
        crop_margin: int = 8,
    ):
        """Args:

        ratio_map: Tensor (H, W) or (1, 1, H, W)
        sky_counts: Tensor (H, W) of OpenSky counts (used for statistical
        masking)
        crop_margin: Number of boundary bins to strip from the outer edges
        """
        if ratio_map.ndim == 2:
            ratio_map = ratio_map.unsqueeze(0).unsqueeze(0)

        # 1. Bandpass filter to suppress edge gradients and single-pixel spikes
        dog_response = F.conv2d(ratio_map, self.kernel, padding="same")

        # 2. Build spatial validity mask to cut out boundary noise
        H, W = ratio_map.shape[-2], ratio_map.shape[-1]
        spatial_mask = torch.zeros(
            (H, W), dtype=torch.bool, device=ratio_map.device
        )
        spatial_mask[
            crop_margin : H - crop_margin, crop_margin : W - crop_margin
        ] = True

        if sky_counts is not None:
            # Exclude bins with poor statistics
            stat_mask = (sky_counts.squeeze() > 5.0) & spatial_mask
            flat_dog = dog_response.squeeze()[stat_mask]
        else:
            flat_dog = dog_response.squeeze()[spatial_mask]

        # 3. Standardize response over the valid interior
        mu = torch.mean(flat_dog)
        sigma = torch.std(flat_dog) + 1e-6

        # 4. Smooth, differentiable peak extraction via LogSumExp
        # SoftMax temperature controls peak sharpness
        peak = self.temperature * torch.logsumexp(
            (flat_dog - mu) / (sigma * self.temperature), dim=0
        )

        # Minimize negative peak response
        return -peak


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


def prefilter_dataset(
    data: np.Array,
    det_search_bbox: tuple[float, float, float, float],
    angle_bounds: float,
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
        & (np.arctan(np.sqrt(data[:,3]**2+data[:,4]**2)/-data[:,5]) <= angle_bounds)
    )
    data_masked = data[mask]
    return torch.from_numpy(data_masked).clone()  # Clone frees up unreferenced slices


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


def prepare_projected_tracks(
    data: np.ndarray,
    det_search_bbox: tuple[float, float, float, float],
    angle_bounds: float,
    margin: float = 200.0,
    detSizeZ: float = 40.0,
    Z: float = 800.0,
    xrange: tuple[float, float] = (-1500.,1500.),
    nbins: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-filters and pre-projects rays on CPU to avoid computing projection math
    inside the GPU autograd loop.
    Returns:
        track_pos: (N, 4) -> [x_top, y_top, x_bot, y_bot] (float32 on CPU)
        proj_idx:  (N, 4) -> [y0_c, x0_c, y1_c, x1_c] (int64 on CPU)
        interp_w:  (N, 4) -> [w00, w01, w10, w11] (float32 on CPU)
    """
    xmin, xmax, ymin, ymax = det_search_bbox
    dz = data[:, 5]
    theta = np.arctan(np.sqrt(data[:, 3] ** 2 + data[:, 4] ** 2) / -dz)

    # Prefilter the dataset
    mask = (
        (data[:, 0] >= xmin - margin)
        & (data[:, 0] <= xmax + margin)
        & (data[:, 1] >= ymin - margin)
        & (data[:, 1] <= ymax + margin)
        & (theta <= angle_bounds)
    )
    d = data[mask]

    # Precalculate geometric tracks
    # x_proj, y_proj: coordinates of muon at chosen Z value
    # x_top, y_top: coordinates of muon at Z = z_detector
    # x_bot, y_bot: coordinates of muon at Z = z_detector - detSizeZ
    dz = d[:, 5]
    l = (Z - d[:, 2]) / dz
    x_proj = d[:, 0] + l * d[:, 3]
    y_proj = d[:, 1] + l * d[:, 4]

    x_top, y_top = d[:, 0], d[:, 1]
    x_bot = d[:, 0] - (detSizeZ / dz) * d[:, 3]
    y_bot = d[:, 1] - (detSizeZ / dz) * d[:, 4]

    track_pos = torch.from_numpy(
        np.column_stack([x_top, y_top, x_bot, y_bot]).astype(np.float32)
    )

    # Precalculate static 2D histogram splatting weights and indices
    x_norm = np.clip((x_proj - (xrange[0])) / (xrange[1]-xrange[0]) * float(nbins-1), 0, nbins-1)
    y_norm = np.clip((y_proj - (xrange[0])) / (xrange[1]-xrange[0]) * float(nbins-1), 0, nbins-1)

    x0 = np.floor(x_norm).astype(np.int64)
    y0 = np.floor(y_norm).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, nbins-1)
    y1 = np.clip(y0 + 1, 0, nbins-1)

    wx1 = (x_norm - x0).astype(np.float32)
    wx0 = (1.0 - wx1).astype(np.float32)
    wy1 = (y_norm - y0).astype(np.float32)
    wy0 = (1.0 - wy1).astype(np.float32)

    w00 = wx0 * wy0
    w01 = wx0 * wy1
    w10 = wx1 * wy0
    w11 = wx1 * wy1

    proj_idx = torch.from_numpy(np.column_stack([y0, x0, y1, x1]))
    interp_w = torch.from_numpy(np.column_stack([w00, w01, w10, w11]))

    return track_pos, proj_idx, interp_w


def project_and_accumulate_streaming(
    track_pos: torch.Tensor,
    proj_idx: torch.Tensor,
    interp_w: torch.Tensor,
    detPos: torch.Tensor,
    detSize: float = 100.0,
    temp: float = 2.0,
    chunk_size: int = 150_000,
) -> torch.Tensor:
    """Streams data from CPU RAM to GPU chunk by chunk."""
    device = detPos.device
    H_bins, W_bins = 50, 50
    hist_acc = torch.zeros(
        (H_bins, W_bins), dtype=torch.float32, device=device
    )

    total_rows = track_pos.shape[0]

    for i in range(0, total_rows, chunk_size):
        # 1. Asynchronously send small chunk to GPU
        b_pos = track_pos[i : i + chunk_size].to(device, non_blocking=True)
        b_idx = proj_idx[i : i + chunk_size].to(device, non_blocking=True)
        b_iw = interp_w[i : i + chunk_size].to(device, non_blocking=True)

        x_top, y_top = b_pos[:, 0], b_pos[:, 1]
        x_bot, y_bot = b_pos[:, 2], b_pos[:, 3]

        # 2. Differentiable acceptance weights for all 3 detectors
        weights = torch.zeros(b_pos.shape[0], dtype=torch.float32, device=device)
        for k in range(detPos.shape[0]):
            pos = detPos[k]
            w_top = soft_box_2d(x_top, y_top, pos, detSize, temperature=temp)
            w_bot = soft_box_2d(x_bot, y_bot, pos, detSize, temperature=temp)
            weights = weights + (w_top * w_bot)

        # 3. Accumulate soft-binned histogram
        w00 = weights * b_iw[:, 0]
        w01 = weights * b_iw[:, 1]
        w10 = weights * b_iw[:, 2]
        w11 = weights * b_iw[:, 3]

        y0, x0 = b_idx[:, 0], b_idx[:, 1]
        y1, x1 = b_idx[:, 2], b_idx[:, 3]

        hist_batch = torch.zeros(
            (H_bins, W_bins), dtype=torch.float32, device=device
        )
        hist_batch.index_put_((y0, x0), w00, accumulate=True)
        hist_batch.index_put_((y1, x0), w01, accumulate=True)
        hist_batch.index_put_((y0, x1), w10, accumulate=True)
        hist_batch.index_put_((y1, x1), w11, accumulate=True)

        hist_acc = hist_acc + hist_batch

    return hist_acc

if __name__ == '__main__':
    NEVENTS = 100000000
    LEARNING_RATE = 40.0
    XRANGE = (-1500.0, 1500.0)
    initDetectorPos = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    tunnel_bounds = [-400.0, 400.0, -1000.0, 1000.0]
    angle_bounds = np.pi / 2.

    # Enable CUDA allocator optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print(">>>> Loading data into memmap...")
    openskyFiles = [
        '/home/ruben/Documents/TunnelInspection/data/outputOpensky_conf2_1-1000_100M.h5'
    ]
    tunnelFiles = [
        '/home/ruben/Documents/TunnelInspection/data/outputTunnel_conf2_1-1000_100M.h5'
    ]

    openskyfull = load_to_memmap(openskyFiles, 'opensky.dat', N=NEVENTS)
    tunnelfull = load_to_memmap(tunnelFiles, 'tunnel.dat', N=NEVENTS)

    print(">>>> Pre-calculating projected geometry on CPU...")
    opensky_pos, opensky_idx, opensky_iw = prepare_projected_tracks(
        openskyfull, tunnel_bounds, angle_bounds, Z=800.0
    )
    tunnel_pos, tunnel_idx, tunnel_iw = prepare_projected_tracks(
        tunnelfull, tunnel_bounds, angle_bounds, Z=800.0
    )

    # Pin memory for fast CPU -> GPU async streaming transfers
    tunnel_pos, tunnel_idx, tunnel_iw = (
        tunnel_pos.pin_memory(),
        tunnel_idx.pin_memory(),
        tunnel_iw.pin_memory(),
    )
    opensky_pos, opensky_idx, opensky_iw = (
        opensky_pos.pin_memory(),
        opensky_idx.pin_memory(),
        opensky_iw.pin_memory(),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    detector_pos = torch.nn.Parameter(
        torch.tensor(
            initDetectorPos,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
    )
    optimizer = torch.optim.Adam([detector_pos], lr=LEARNING_RATE)
    blind_loss = ChamberBlindDetectorLoss(kernel_size=8, sigma=2.0).to(device)
    blind_loss_2 = ChamberDoGLoss().to(device)

    print(">>>> Starting optimization loop...")
    for epoch in range(50):
        optimizer.zero_grad(set_to_none=True)

        H_tunnel = project_and_accumulate_streaming(
            tunnel_pos,
            tunnel_idx,
            tunnel_iw,
            detector_pos,
            detSize=100.0,
            temp=2.0,
            chunk_size=150_000,
        )
        H_sky = project_and_accumulate_streaming(
            opensky_pos,
            opensky_idx,
            opensky_iw,
            detector_pos,
            detSize=100.0,
            temp=2.0,
            chunk_size=150_000,
        )

        ratio_map = H_tunnel / (H_sky + 1e-3)

        loss = blind_loss_2(ratio_map, sky_counts = H_sky)
        loss.backward()
        optimizer.step()

        # Enforce search boundaries
        detPos_boundaries = [-400.0, 400.0, -600.0, 600.0]
        with torch.no_grad():
            detector_pos[:, 0].clamp_(
                min=detPos_boundaries[0] + 50.0, max=detPos_boundaries[1] - 50.0
            )
            detector_pos[:, 1].clamp_(
                min=detPos_boundaries[2] + 50.0, max=detPos_boundaries[3] - 50.0
            )

        print(
            f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | "
            f"Det 0: {detector_pos[0].detach().cpu().numpy().round(1)} | "
            f"Det 1: {detector_pos[1].detach().cpu().numpy().round(1)} | "
            f"Det 2: {detector_pos[2].detach().cpu().numpy().round(1)}"
        )

        # Plot epoch ratio hist
        fig,ax = plt.subplots(1, 3, figsize = (24, 6))
        xedges = np.linspace(XRANGE[0], XRANGE[1], 51)
        yedges = np.linspace(XRANGE[0], XRANGE[1], 51)
        pc = ax[0].pcolorfast(xedges, yedges, ratio_map.detach().cpu().numpy().T)
        fig.colorbar(pc, ax=ax[0])
        smoothed = F.conv2d(ratio_map.unsqueeze(0).unsqueeze(0), blind_loss_2.kernel, padding="same")
        pc1 = ax[1].pcolorfast(xedges, yedges, np.array(smoothed[0][0].detach().cpu().numpy()).T)
        fig.colorbar(pc1, ax=ax[1])
        detPos1 = detector_pos[0].detach().cpu().numpy()
        detPos2 = detector_pos[1].detach().cpu().numpy()
        detPos3 = detector_pos[2].detach().cpu().numpy()
        det1 = patches.Rectangle((detPos1[0]-50,detPos1[1]-50), 100, 100, edgecolor='red', facecolor='none', linewidth=2)
        det2 = patches.Rectangle((detPos2[0]-50,detPos2[1]-50), 100, 100, edgecolor='red', facecolor='none', linewidth=2)
        det3 = patches.Rectangle((detPos3[0]-50,detPos3[1]-50), 100, 100, edgecolor='red', facecolor='none', linewidth=2)
        ax[2].add_patch(det1)
        ax[2].add_patch(det2)
        ax[2].add_patch(det3)
        ax[2].set_ylim(-500,500)
        ax[2].set_xlim(-500,500)
        plt.savefig(f'plots/epoch_{epoch}.png')
        plt.close()


        
'''if __name__ == '__main__':
    NEVENTS = 60000000
    LEARNING_RATE = 25.
    XRANGE = (-1500.,1500.)
    initDetectorPos = [[-200.0, -200.0], [-0.0, 0.0], [100.0, 50.0]]

    print(">>>> Loading data into memmap...")
    #openskyFiles = ['/home/ruben/Documents/TunnelInspection/data/outputOpensky_v3_249p8_compressed.h5']
    #tunnelFiles  = ['/home/ruben/Documents/TunnelInspection/data/outputTunnel_v4_596p6_compressed.h5']
    openskyFiles = ['/home/ruben/Documents/TunnelInspection/data/outputOpensky_conf2_1-1000_100M.h5']
    tunnelFiles = ['/home/ruben/Documents/TunnelInspection/data/outputTunnel_conf2_1-1000_100M.h5']
    

    openskyfull = load_to_memmap(openskyFiles, 'opensky.dat', N=NEVENTS)
    tunnelfull  = load_to_memmap(tunnelFiles, 'tunnel.dat', N=NEVENTS)

    # 1. Pre-filter broadly once on disk/CPU before converting to PyTorch tensors
    tunnel_bounds = [-400.0, 400.0, -1000.0, 1000.0]
    angle_bounds = 4.*np.pi/9.
    print(">>>> Pre-filtering dataset around region of interest...")
    openskyTensor = prefilter_dataset(openskyfull, tunnel_bounds, angle_bounds, margin=200.0)
    tunnelTensor  = prefilter_dataset(tunnelfull,  tunnel_bounds, angle_bounds, margin=200.0)
    print(f"Filtered Tunnel shape: {tunnelTensor.shape}, OpenSky shape: {openskyTensor.shape}")

    fig, axs = plt.subplots(2, 3, figsize = (12, 8), tight_layout=True)
    # We can set the number of bins with the *bins* keyword argument.
    n_bins = 100
    xrange = (-1000,1000)
    axs[0][0].hist(openskyTensor[:,0], bins=n_bins, range=xrange, histtype='step');
    axs[0][0].set_xlabel("x (cm)")
    axs[0][1].hist(openskyTensor[:,1], bins=n_bins, range=xrange, histtype='step');
    axs[0][1].set_xlabel("y (cm)")
    axs[0][2].hist(openskyTensor[:,2], bins=n_bins, range=(-730,-729), histtype='step');
    axs[0][2].set_xlabel("z (cm)")
    axs[1][0].hist(tunnelTensor[:,0], bins=n_bins, range=xrange, histtype='step');
    axs[1][0].set_xlabel("x (cm)")
    axs[1][1].hist(tunnelTensor[:,1], bins=n_bins, range=xrange, histtype='step');
    axs[1][1].set_xlabel("y (cm)")
    axs[1][2].hist(tunnelTensor[:,2], bins=n_bins, range=(-730,-729), histtype='step');
    axs[1][2].set_xlabel("z (cm)")
    plt.savefig("plots/positions.png")

    xrange = [-1.,1.]
    fig, axs = plt.subplots(2, 3, figsize = (12, 8), tight_layout=True)
    # We can set the number of bins with the *bins* keyword argument.
    n_bins = 100
    xrange = (-1.,1.)
    axs[0][0].hist(openskyTensor[:,3], bins=n_bins, range=xrange, histtype='step');
    axs[0][0].set_xlabel("vx")
    axs[0][1].hist(openskyTensor[:,4], bins=n_bins, range=xrange, histtype='step');
    axs[0][1].set_xlabel("vy")
    axs[0][2].hist(openskyTensor[:,5], bins=n_bins, range=xrange, histtype='step');
    axs[0][2].set_xlabel("vz")
    axs[1][0].hist(tunnelTensor[:,3], bins=n_bins, range=xrange, histtype='step');
    axs[1][0].set_xlabel("vx")
    axs[1][1].hist(tunnelTensor[:,4], bins=n_bins, range=xrange, histtype='step');
    axs[1][1].set_xlabel("vy")
    axs[1][2].hist(tunnelTensor[:,5], bins=n_bins, range=xrange, histtype='step');
    axs[1][2].set_xlabel("vz")
    plt.savefig("plots/directions.png")

    xrange = [-1.,1.]
    fig, axs = plt.subplots(2, 1, figsize = (4, 8), tight_layout=True)
    # We can set the number of bins with the *bins* keyword argument.
    n_bins = 100
    xrange = (0.,1.6)
    theta_opensky = torch.arctan(torch.sqrt(openskyTensor[:,3]**2+openskyTensor[:,4]**2)/-openskyTensor[:,5])
    theta_tunnel = torch.arctan(torch.sqrt(tunnelTensor[:,3]**2+tunnelTensor[:,4]**2)/-tunnelTensor[:,5])
    axs[0].hist(theta_opensky, bins=n_bins, range=xrange, histtype='step');
    axs[0].set_xlabel("theta")
    axs[1].hist(theta_tunnel, bins=n_bins, range=xrange, histtype='step');
    axs[1].set_xlabel("theta")
    plt.savefig("plots/theta.png")

    # Move to GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    tunnelTensor = tunnelTensor.to(device)
    openskyTensor = openskyTensor.to(device)

    # 2. Setup parameters and optimizer
    detector_pos = torch.nn.Parameter(
        torch.tensor(initDetectorPos, dtype=torch.float32, device=device, requires_grad=True)
    )
    optimizer = torch.optim.Adam([detector_pos], lr=LEARNING_RATE)

    hist_builder = DifferentiableHistogram2D(
        bins=(50, 50), x_range=XRANGE, y_range=XRANGE
    ).to(device)
    blind_loss = ChamberBlindDetectorLoss(kernel_size=15, sigma=3.0).to(device)

    # 3. Iterative Loop
    for epoch in range(50):
        optimizer.zero_grad()

        # Differentiable forward pass accumulated in small chunks
        H_tunnel = project_muons_chunked(tunnelTensor, detector_pos, hist_builder, chunk_size=100000, Z=800.)
        H_sky    = project_muons_chunked(openskyTensor, detector_pos, hist_builder, chunk_size=100000, Z=800.)

        ratio_map = H_tunnel / (H_sky + 1e-3)
        #valid_mask = (ratio_map > 0.9*torch.max(ratio_map)).float()
        #print(valid_mask)
        #print(H_tunnel, torch.max(H_tunnel))
        #print(H_sky, torch.max(H_sky))

        # Plot epoch ratio hist
        fig,ax = plt.subplots(2, 1, figsize = (8, 12))
        xedges = np.linspace(XRANGE[0], XRANGE[1], 51)
        yedges = np.linspace(XRANGE[0], XRANGE[1], 51)
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
        detPos_boundaries = [-400.,400.,-600.,600.]
        with torch.no_grad():
            detector_pos[:, 0].clamp_(min=detPos_boundaries[0]+50., max=detPos_boundaries[1]-50.)
            detector_pos[:, 1].clamp_(min=detPos_boundaries[2]+50., max=detPos_boundaries[3]-50.)

        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")
        print(f"                      Detector 0 position: {detector_pos[0].detach().cpu().numpy()}")
        print(f"                      Detector 1 position: {detector_pos[1].detach().cpu().numpy()}")
        print(f"                      Detector 2 position: {detector_pos[2].detach().cpu().numpy()}")'''
