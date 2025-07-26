import math
import os
import numpy as np
import torch
from torch import nn
from gs.core.BaseCamera import BaseCamera
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gs.core.BasePointCloud import BasePointCloud
from gs.helpers.math import inverse_sigmoid
from gs.helpers.spherical_harmonics import rgb_to_sh
from gs.helpers.system import mkdir_p
from gs.helpers.transforms import build_covariance_from_scaling_rotation
from simple_knn._C import distCUDA2
from plyfile import PlyData, PlyElement
from gs.helpers.aqua import load_planes
import torch.nn.functional as F

import json
from pathlib import Path

from gs.helpers.aqua import apply_refraction_to_gaussian

# === new helper to load planes.json once ===

_PLANES = load_planes()


class GaussianModel(nn.Module):
    """
    Base class for Gaussian models, represents collection of Gaussians with spherical harmonics coefficients that can be rendered.
    """

    background_color: torch.Tensor
    max_radii2D: torch.Tensor
    _gradient_accumulator: torch.Tensor
    _gradient_accumulator_denominator: torch.Tensor
    radii: torch.Tensor

    # Basic functionality

    def __init__(
            self,
            # Neccessary parameters
            positions: torch.Tensor,  # Position of Gaussians. (N, 3), N = number of Gaussians
            sh_coefficients: torch.Tensor,  # Spherical harmonics coefficients. (N, sh_degree)
            rotations: torch.Tensor,  # Rotation of Gaussians. (N, 3)
            scales: torch.Tensor,  # Scale of Gaussians. (N, 3)
            opacities: torch.Tensor,  # Opacity of Gaussians. (N, 1)
            # Other parameters
            sh_degree: int = 4,  # Degree of spherical harmonics
            background_color: torch.Tensor = torch.tensor([0, 0, 0], dtype=torch.float32),  # Background color
    ):
        super().__init__()
        # Gaussian parameters defining geometry and appearance to be optimized.
        self.positions = nn.Parameter(positions).float()
        self.sh_coefficients_0 = nn.Parameter(sh_coefficients[:, :1, :])
        self.sh_coefficients_rest = nn.Parameter(sh_coefficients[:, 1:, :])
        # ensure everything coming off self is in float32

        self.scales = self.scales.float()
        self.rotations = self.rotations.float()
        self.opacities = self.opacities.float()

        # Intermediate variables
        self.sh_degree = sh_degree
        background_color = background_color
        self.viewspace_points = None

        # Set activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # Stats for densification
        # Initialize _gradient_accumulator to be of shape (N, 1) where N is the number of Gaussians
        _gradient_accumulator = torch.zeros((positions.shape[0], 1))
        _gradient_accumulator_denominator = torch.zeros((positions.shape[0], 1))
        max_radii2D = torch.zeros((positions.shape[0], 1))

        self.register_buffer("background_color", background_color, persistent=True)
        self.register_buffer("_gradient_accumulator", _gradient_accumulator, persistent=True)
        self.register_buffer("_gradient_accumulator_denominator", _gradient_accumulator_denominator, persistent=True)
        self.register_buffer("max_radii2D", max_radii2D, persistent=True)

        self.radii = None

    def forward(self, camera: BaseCamera, active_sh_degree: int = None):
        """
        Simplified forward: only refract Gaussian means (no Jacobian/covariance warping).

        1) Compute refracted means via batched Snell's law.
        2) Keep original packed covariances.
        3) Rasterize in chunks to stay GPU-friendly.
        """
        import torch.nn.functional as F
        device = self.positions.device

        # -- 1) Select active SH degree
        if active_sh_degree is None:
            active_sh_degree = self.sh_degree
        else:
            active_sh_degree = min(active_sh_degree, self.sh_degree)

        # -- 2) Prepare viewspace_points for densification stats
        self.viewspace_points = torch.zeros_like(
            self.positions, requires_grad=True, device=device
        )
        try:
            self.viewspace_points.retain_grad()
        except:
            pass

        # -- 3) Build rasterizer settings
        tanfovx = math.tan(camera.fov_x * 0.5)
        tanfovy = math.tan(camera.fov_y * 0.5)
        rs = GaussianRasterizationSettings(
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            viewmatrix=camera.world_view_transform.to(device).float(),
            projmatrix=camera.full_proj_transform.to(device).float(),
            campos=camera.camera_center.to(device).float(),
            sh_degree=active_sh_degree,
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            bg=torch.zeros(3, device=device, dtype=torch.float32),
            scale_modifier=1.0,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=rs)

        # -- 4) Get original packed covariances (N,6)
        cov_symm = build_covariance_from_scaling_rotation(
            self.scales.to(device).float(),
            rs.scale_modifier,
            self.rotations.to(device).float()
        ).float()

        # -- 5) Initialize rays
        N = self.positions.shape[0]
        O0 = rs.campos.view(1, 3)  # camera center
        O = O0.expand(N, 3).clone()  # origins per Gaussian

        means = self.positions  # (N,3)
        D = F.normalize(means - O, dim=1)  # directions
        dist = torch.norm(means - O, dim=1, keepdim=True)
        n = torch.ones(N, device=device, dtype=torch.float32)

        # -- 6) Refract means through each plane
        for p in _PLANES:
            origin = torch.tensor(p['origin'], device=device, dtype=torch.float32).view(1, 3)
            normal = torch.tensor(p['normal'], device=device, dtype=torch.float32).view(1, 3)
            n2 = torch.tensor(p['n2'], device=device, dtype=torch.float32)

            # intersect ray with plane
            denom = (D * normal).sum(dim=1)
            mask = denom.abs() > 1e-6
            t = ((origin - O) * normal).sum(dim=1) / denom
            mask &= t > 0
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=False).squeeze(1)
            Dm = D[idx]
            Nm = normal.expand_as(Dm)
            n1 = n[idx]

            # Snell's law
            eta = (n1 / n2).unsqueeze(1)
            cosi = -(Nm * Dm).sum(dim=1, keepdim=True)
            k = 1 - eta ** 2 * (1 - cosi ** 2)
            sqrtk = torch.sqrt(k.clamp(min=0))
            Dp = eta * Dm + (eta * cosi - sqrtk) * Nm

            # advance origin
            hit_pt = O[idx] + Dm * t[idx].unsqueeze(1)
            O[idx] = hit_pt + 1e-4 * Dp
            D[idx] = Dp
            n[idx] = n2

        # -- 7) Warp means only
        means_warped = O + D * dist  # (N,3)

        # -- 8) In-place update of parameters (avoid leaks)
        with torch.no_grad():
            self.positions.copy_(means_warped)

        # -- 9) Chunked rasterization
        H, W = int(camera.image_height), int(camera.image_width)
        accum_image = torch.zeros((3, H, W), device=device)
        accum_radii = torch.zeros((N, 1), device=device)
        batch_size = 30000
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            img_chunk, rad_chunk = rasterizer(
                means3D=means_warped[i:j],
                means2D=self.viewspace_points[i:j],
                shs=self.sh_coefficients[i:j],
                opacities=self.opacity_activation(self.opacities[i:j]),
                scales=None,
                rotations=None,
                cov3Ds_precomp=cov_symm[i:j]
            )
            accum_image += img_chunk
            accum_radii[i:j] = rad_chunk.view(-1, 1)

        self.radii = accum_radii
        return accum_image.clamp(0.0, 1.0)

    def backprop_stats(self):
        """
        Backpropagate stats for densification. Called after loss.backward().
        """
        if self.viewspace_points is None:
            raise ValueError(
                "viewspace_points is not initialized. Please call forward() before calling update_stats()."
            )
        if self.radii is None:
            raise ValueError(
                "radii is not initialized. Please call forward() before calling update_stats()."
            )

        # mask: (N,1), 1 for visible Gaussians, 0 otherwise
        mask = (self.radii > 0).float()  # still float32, GPU

        # 1) accumulate gradient magnitudes
        if self.viewspace_points.grad is not None:
            # grad_xy: (N,2)
            grad_xy = self.viewspace_points.grad[:, :2]
            # grad_norm: (N,1)
            grad_norm = torch.norm(grad_xy, dim=-1, keepdim=True)
            # weighted add: only visible ones contribute
            self._gradient_accumulator += grad_norm * mask
            self._gradient_accumulator_denominator += mask

        # 2) update max radii
        # new_max = elementwise max of old vs current radii
        new_max = torch.max(self.max_radii2D, self.radii)
        # select per-Gaussian: if visible, take new_max, else keep old
        self.max_radii2D = torch.where(mask.bool(), new_max, self.max_radii2D)

    @property
    def sh_coefficients(self):
        return torch.cat([self.sh_coefficients_0, self.sh_coefficients_rest], dim=1)

    @sh_coefficients.setter
    def sh_coefficients(self, value):
        self.sh_coefficients_0 = value[:, :1, :]
        self.sh_coefficients_rest = value[:, 1:, :]

    @property
    def mean_gradient_magnitude(self):
        m = self._gradient_accumulator / self._gradient_accumulator_denominator
        # Count the number of NaN values in either the numerator or the denominator
        accum_nan = torch.isnan(self._gradient_accumulator).sum()
        denom_nan = torch.isnan(self._gradient_accumulator_denominator).sum()
        if accum_nan > 0:
            print(f"Warning: _gradient_accumulator contains {accum_nan} NaN values.")
        if denom_nan > 0:
            print(f"Warning: _gradient_accumulator_denominator contains {denom_nan} NaN values.")
        m[torch.isnan(m)] = 0
        return m

    @staticmethod
    def from_point_cloud(pointcloud: BasePointCloud, sh_degree: int = 3,
                         background_color: torch.Tensor = torch.tensor([0, 0, 0], dtype=torch.float32),
                         constant_scale: float = None):
        """
        Create GaussianModel from PointCloud. This is useful for converting PointClouds to GaussianModels, which can be rendered using rasterizer.
        """
        # Initialize positions
        positions = torch.tensor(np.asarray(pointcloud.points)).float()

        # Initialize spherical harmonics coefficients from RGB colors
        sh_0 = rgb_to_sh(torch.tensor(np.asarray(pointcloud.colors)).float())
        sh_coefficients = torch.zeros((sh_0.shape[0], 3, (sh_degree + 1) ** 2)).float()
        sh_coefficients[:, :3, 0] = sh_0
        sh_coefficients = sh_coefficients.transpose(1, 2)

        # Initialize scale
        if constant_scale is not None:
            scales = torch.log(torch.tensor([constant_scale], dtype=torch.float).repeat(positions.shape[0], 3))
        else:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pointcloud.points)).float().cuda()),
                                    0.0000001)  # Calculate squared distance between points
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # Initialize rotation
        rotations = torch.zeros((positions.shape[0], 4), device="cuda")  # Create a tensor: (N, 4) for quaternions
        rotations[:, 0] = 1  # Set the first column to 1, such that the rotation is identity

        # Initialize opacity
        # opacities = inverse_sigmoid(0.1 * torch.ones((positions.shape[0], 1), dtype=torch.float, device="cuda")) # Set opacity to 0.1
        opacities = inverse_sigmoid(
            0.1 * torch.ones((positions.shape[0], 1), dtype=torch.float, device="cuda"))  # Set opacity to 0.1

        return GaussianModel(
            positions=positions,
            sh_coefficients=sh_coefficients,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            sh_degree=sh_degree,
            background_color=background_color
        )

    # Convenience functions

    def clone(self):
        """
        Clone the GaussianModel.
        """
        return GaussianModel(
            positions=self.positions.clone(),
            sh_coefficients=self.sh_coefficients.clone(),
            rotations=self.rotations.clone(),
            scales=self.scales.clone(),
            opacities=self.opacities.clone(),
            sh_degree=self.sh_degree,
            background_color=self.background_color.clone()
        )

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx):
        """
        Allows indexing to subset Gaussian models using boolean masks or indices.
        """
        # Ensure idx is a boolean mask or slice to properly subset tensors
        if isinstance(idx, slice) or (isinstance(idx, torch.Tensor) and idx.dtype == torch.bool):
            new_positions = self.positions[idx]
            new_sh_coefficients_0 = self.sh_coefficients_0[idx]
            new_sh_coefficients_rest = self.sh_coefficients_rest[idx]
            new_rotations = self.rotations[idx]
            new_scales = self.scales[idx]
            new_opacities = self.opacities[idx]

            return GaussianModel(
                positions=new_positions,
                sh_coefficients=torch.cat([new_sh_coefficients_0, new_sh_coefficients_rest], dim=1),
                rotations=new_rotations,
                scales=new_scales,
                opacities=new_opacities,
                sh_degree=self.sh_degree,
                background_color=self.background_color
            )
        else:
            raise TypeError("Indexing with type {} not supported".format(type(idx)))

    def __setitem__(self, idx, new_model):
        """
        Allows setting values for a subset of Gaussians in the model using indexing notation.
        """
        if not isinstance(new_model, GaussianModel):
            raise ValueError("Assigned value must be an instance of GaussianModel.")

        self.positions[idx] = new_model.positions
        self.sh_coefficients_0[idx] = new_model.sh_coefficients_0
        self.sh_coefficients_rest[idx] = new_model.sh_coefficients_rest
        self.rotations[idx] = new_model.rotations
        self.scales[idx] = new_model.scales
        self.opacities[idx] = new_model.opacities

    @staticmethod
    def concatenate(*models: "GaussianModel"):
        """
        Concatenates multiple GaussianModel instances into a single model.
        """
        if not all(isinstance(model, GaussianModel) for model in models):
            raise ValueError("All items to concatenate must be instances of GaussianModel.")

        # Check for consistent sh_degree and background_color across all models
        if len(set(model.sh_degree for model in models)) > 1:
            raise ValueError("All models must have the same spherical harmonics degree.")
        if len(set(tuple(model.background_color.tolist()) for model in models)) > 1:
            raise ValueError("All models must have the same background color.")

        # Concatenate all parameters
        concatenated_positions = torch.cat([model.positions for model in models], dim=0)
        concatenated_sh_coefficients_0 = torch.cat([model.sh_coefficients_0 for model in models], dim=0)
        concatenated_sh_coefficients_rest = torch.cat([model.sh_coefficients_rest for model in models], dim=0)
        concatenated_rotations = torch.cat([model.rotations for model in models], dim=0)
        concatenated_scales = torch.cat([model.scales for model in models], dim=0)
        concatenated_opacities = torch.cat([model.opacities for model in models], dim=0)

        # Create a new GaussianModel instance with concatenated parameters
        new_model = GaussianModel(
            positions=concatenated_positions,
            sh_coefficients=torch.cat([concatenated_sh_coefficients_0, concatenated_sh_coefficients_rest], dim=1),
            rotations=concatenated_rotations,
            scales=concatenated_scales,
            opacities=concatenated_opacities,
            sh_degree=models[0].sh_degree,  # Assuming all have the same sh_degree
            background_color=models[0].background_color  # Assuming all have the same background color
        )
        return new_model

    def concatenate(self, *models: "GaussianModel"):
        """
        Concatenates multiple GaussianModel instances into the current model.
        """
        if not all(isinstance(model, GaussianModel) for model in models):
            raise ValueError("All items to concatenate must be instances of GaussianModel.")

        # Check for consistent sh_degree and background_color across all models
        if len(set(model.sh_degree for model in models)) > 1:
            raise ValueError("All models must have the same spherical harmonics degree.")
        if len(set(tuple(model.background_color.tolist()) for model in models)) > 1:
            raise ValueError("All models must have the same background color.")

        # Concatenate all parameters
        concatenated_positions = torch.cat([model.positions for model in models], dim=0)
        concatenated_sh_coefficients_0 = torch.cat([model.sh_coefficients_0 for model in models], dim=0)
        concatenated_sh_coefficients_rest = torch.cat([model.sh_coefficients_rest for model in models], dim=0)
        concatenated_rotations = torch.cat([model.rotations for model in models], dim=0)
        concatenated_scales = torch.cat([model.scales for model in models], dim=0)
        concatenated_opacities = torch.cat([model.opacities for model in models], dim=0)

        # Update current model with concatenated parameters
        self.positions = concatenated_positions
        self.sh_coefficients_0 = concatenated_sh_coefficients_0
        self.sh_coefficients_rest = concatenated_sh_coefficients_rest
        self.rotations = concatenated_rotations
        self.scales = concatenated_scales
        self.opacities = concatenated_opacities

    def save_ply(self, filename: str):
        """
        Save the GaussianModel to a PLY file. Follows original implementation.
        Useful for saving for visualization, but to save with the ability to resume training, better use PyTorch's save/load functionality.
        """
        # Make sure the directory to save the file exists
        mkdir_p(os.path.dirname(filename))

        # print(f"Shapes: positions={self.positions.shape}, sh_coefficients_0={self.sh_coefficients_0.shape}, sh_coefficients_rest={self.sh_coefficients_rest.shape}, scales={self.scales.shape}, rotations={self.rotations.shape}, opacities={self.opacities.shape}")

        # Detach parameters and move to CPU to prepare for saving
        positions = self.positions.detach().cpu().numpy()
        normals = np.zeros_like(positions)
        rotations = self.rotations.detach().cpu().numpy()
        scales = self.scales.detach().cpu().numpy()
        opacities = self.opacities.detach().cpu().numpy()
        sh_coefficients_0 = self.sh_coefficients_0.detach().cpu().squeeze(1).numpy()
        sh_coefficients_rest = self.sh_coefficients_rest.detach().cpu().reshape(self.sh_coefficients_rest.shape[0],
                                                                                -1).numpy()

        # Prepare PLY attributes (order matters!)
        basic_attribs = ['x', 'y', 'z', 'nx', 'ny', 'nz']  # Position and normal attributes
        sh_coefficients_0_attribs = [f'f_dc_{i}' for i in range(
            self.sh_coefficients_0.shape[1] * self.sh_coefficients_0.shape[2])]  # degrees * 3 channels
        sh_coefficients_rest_attribs = [f'f_rest_{i}' for i in
                                        range(self.sh_coefficients_rest.shape[1] * self.sh_coefficients_rest.shape[2])]
        opacity_attribs = ['opacity']
        scaling_attribs = [f'scale_{i}' for i in range(self.scales.shape[1])]
        rotation_attribs = [f'rot_{i}' for i in range(self.rotations.shape[1])]
        attribs = basic_attribs + sh_coefficients_0_attribs + sh_coefficients_rest_attribs + opacity_attribs + scaling_attribs + rotation_attribs
        dtype = [(attribute, 'f4') for attribute in attribs]  # Save all as float32 (4 bytes)

        # Save the attributes to a PLY file
        elements = np.empty(len(self), dtype=dtype)
        attributes = np.concatenate(
            (positions, normals, sh_coefficients_0, sh_coefficients_rest, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        element = PlyElement.describe(elements, 'vertex')
        PlyData([element]).write(filename)

    @staticmethod
    def from_ply(filename: str, sh_channels: int = 3):
        plydata = PlyData.read(filename)

        # Extract positions
        positions = np.stack((
            np.asarray(plydata['vertex']['x']),
            np.asarray(plydata['vertex']['y']),
            np.asarray(plydata['vertex']['z']),
        ), axis=1)

        # Extract opacities
        opacities = np.asarray(plydata['vertex']['opacity']).reshape(-1, 1)

        # Extract spherical harmonics coefficients for DC term
        sh_coefficients_0 = []
        for i in range(sh_channels):
            sh_coefficients_0.append(np.asarray(plydata['vertex'][f'f_dc_{i}']))
        sh_coefficients_0 = np.stack(sh_coefficients_0, axis=1).reshape(-1, 1, sh_channels)

        # Extract spherical harmonics coefficients for the rest
        sh_coefficients_rest = []
        extra_sh_names = sorted([p.name for p in plydata['vertex'].properties if p.name.startswith('f_rest_')],
                                key=lambda x: int(x.split('_')[-1]))
        for name in extra_sh_names:
            sh_coefficients_rest.append(np.asarray(plydata['vertex'][name]))
        num_gaussians = len(opacities)
        sh_coefficients_rest = np.stack(sh_coefficients_rest, axis=1).reshape(num_gaussians, -1, sh_channels)

        # Extract scales and rotations
        scale_names = sorted([p.name for p in plydata['vertex'].properties if p.name.startswith("scale_")],
                             key=lambda x: int(x.split('_')[-1]))
        scales = np.stack([np.asarray(plydata['vertex'][name]) for name in scale_names], axis=1)

        rot_names = sorted([p.name for p in plydata['vertex'].properties if p.name.startswith("rot")],
                           key=lambda x: int(x.split('_')[-1]))
        rotations = np.stack([np.asarray(plydata['vertex'][name]) for name in rot_names], axis=1)

        # print(f"Shapes: positions={positions.shape}, sh_coefficients_0={sh_coefficients_0.shape}, sh_coefficients_rest={sh_coefficients_rest.shape}, scales={scales.shape}, rotations={rotations.shape}, opacities={opacities.shape}")

        # Create a new GaussianModel instance with the loaded parameters
        gaussian_model = GaussianModel(
            positions=torch.tensor(positions, dtype=torch.float32).cuda(),
            sh_coefficients=torch.cat([
                torch.tensor(sh_coefficients_0, dtype=torch.float32).cuda(),
                torch.tensor(sh_coefficients_rest, dtype=torch.float32).cuda()
            ], dim=1),
            scales=torch.tensor(scales, dtype=torch.float32).cuda(),
            rotations=torch.tensor(rotations, dtype=torch.float32).cuda(),
            opacities=torch.tensor(opacities, dtype=torch.float32).cuda(),
            sh_degree=sh_channels - 1
        )

        return gaussian_model