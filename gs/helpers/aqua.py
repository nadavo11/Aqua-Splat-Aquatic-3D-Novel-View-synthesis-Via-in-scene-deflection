import numpy as np
import json
from pathlib import Path

def load_planes(json_path="../planes.json"):
    data = json.loads(Path(json_path).read_text())
    # convert to numpy for speed
    return [
      {
        "origin": np.array(p["origin"], dtype=float),
        "normal": np.array(p["normal"], dtype=float),
        "n1": p["refractive_indices"]["from"],
        "n2": p["refractive_indices"]["to"],
        "extents": p["extents"]
      }
      for p in data
    ]
_PLANES = load_planes()
def refract_dir(I, N, n1, n2):
    # I, N: unit vectors in world—both numpy
    η = n1 / n2
    cosi = -np.dot(N, I)
    k = 1 - η*η*(1 - cosi*cosi)
    if k < 0:
        return None  # TIR (shouldn’t happen if n1<n2)
    return η*I + (η*cosi - np.sqrt(k))*N

def _symm_to_mat(symm6):
    # (c00, c01, c02, c11, c12, c22) → full 3×3
    c00, c01, c02, c11, c12, c22 = symm6
    return np.array([
        [c00, c01, c02],
        [c01, c11, c12],
        [c02, c12, c22],
    ], dtype=float)

def _mat_to_symm(mat3):
    # full 3×3 → (c00, c01, c02, c11, c12, c22)
    return np.array([
        mat3[0,0], mat3[0,1], mat3[0,2],
                   mat3[1,1], mat3[1,2],
                              mat3[2,2],
    ], dtype=float)

def apply_refraction_to_gaussian(mean, cov3D, camera, planes=_PLANES):
    """
    mean: numpy (3,), cov3D: numpy (3,3)
    camera: has viewmatrix & projmatrix & camera_center (torch or numpy)
    returns (mean', cov3D') warped through all interfaces
    """

    # unpack if necessary
    packed = (cov3D.ndim == 1 and cov3D.shape[0] == 6)
    cov_mat = _symm_to_mat(cov3D) if packed else cov3D.copy()

    # 1. build ray direction in world space
    #    a) transform mean → clip space → NDC → pupil ray
    #    b) or simpler: ray_dir = normalize(mean - camera.camera_center)
    O = camera.camera_center.cpu().numpy()
    D = mean - O
    D /= np.linalg.norm(D)

    current_n = 1.0  # start in air
    current_O, current_D = O, D.copy()
    J = np.eye(3)   # accumulated Jacobian

    for p in planes:
        N = p["normal"]
        # ray‐plane intersection: t = dot(origin−O, N) / dot(D,N)
        denom = current_D.dot(N)
        if abs(denom) < 1e-6:
            continue
        t = (p["origin"] - current_O).dot(N) / denom
        if t < 0:
            continue
        hit = current_O + current_D * t
        # update Jacobian: for a planar interface, J_new = J_refraction * J_old
        # where J_refraction = ∂D'/∂D = η*I + ...
        η = current_n / p["n2"]
        cosi = -N.dot(current_D)
        k = 1 - η*η*(1 - cosi*cosi)
        Jr = η*np.eye(3) + \
             np.outer((η*cosi - np.sqrt(k))*N - 2*η*np.dot(N,current_D)*N, N)  # approximate
        J = Jr.dot(J)

        # refract direction & step origin
        Dp = refract_dir(current_D, N, current_n, p["n2"])
        current_n = p["n2"]
        current_O = hit + 1e-4 * Dp
        current_D = Dp

    # now propagate mean & covariance
    mean_warped = O + current_D * np.linalg.norm(mean - O)
    cov_warped = J.dot(cov_mat).dot(J.T)
    if packed:
        cov_warped = _mat_to_symm(cov_warped)

    return mean_warped, cov_warped
