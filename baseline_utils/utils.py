
# ------- Imports --------------------------------------------------------------
import os,tempfile, zipfile, requests, imageio, torch
from pathlib import Path
from plyfile import PlyElement

from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.helpers.loss import l1_loss

url = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
ROOT = Path("./data")
DATA_NAME = "tandt"
DATA_DIR = Path(f"./data/{DATA_NAME}/train")
# ------- Prepare dataset ------------------------------------------------------
# We'll download the authors' Tanks&Temples+DeepBlending COLMAP archive and extract only the 'tandt' scene

def get_data_from_web(url = url,
                      root_dir = ROOT,
                      data_name = DATA_NAME,
):

    target_dir = root_dir / data_name
    if not root_dir.exists():
        root_dir.mkdir(parents=True)
    data_dir = target_dir / "train"

    if not data_dir.exists():
        print("Dataset not found – downloading precomputed COLMAP for Tanks&Temples (650 MB)…")
        tmpfile = tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name

        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(tmpfile, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(tmpfile, 'r') as z:
            members = [m for m in z.namelist()]
            z.extractall(path=str(root_dir), members=members)
        os.remove(tmpfile)
