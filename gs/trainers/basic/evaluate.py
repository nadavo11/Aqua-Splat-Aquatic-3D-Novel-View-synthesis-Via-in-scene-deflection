import lpips, imageio, numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pathlib import Path, PurePath
import torch, time, csv
import csv, json, torch, lpips, imageio, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import csv, torch, lpips, imageio, wandb, numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import gc

os.environ["WANDB_SILENT"] = "true"  # hide W&B chatty logs
# (optional) also stop W&B from hijacking your stdout/stderr
os.environ["WANDB_CONSOLE"] = "off"


def eval_model_wandb(model, val_cams, iter_num, run,
                     project="gs-red-truck", lpips_fn=None, device="cuda"):
    if not lpips_fn:
        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    model = model.to(device).eval()

    rows, psnr_all, ssim_all, lp_all = [], [], [], []

    with torch.no_grad():
        for cam in val_cams:
            cam = cam.to(device)
            pred = model.forward(cam, active_sh_degree=model.sh_degree).clamp(0, 1)
            gt = cam.image.float()

            psnr = peak_signal_noise_ratio(gt.cpu().numpy().transpose(1, 2, 0),
                                           pred.cpu().numpy().transpose(1, 2, 0),
                                           data_range=1.0)
            ssim = structural_similarity(gt.cpu().numpy().transpose(1, 2, 0),
                                         pred.cpu().numpy().transpose(1, 2, 0),
                                         channel_axis=2, data_range=1.0)

            pred_tensor = (pred * 2 - 1).unsqueeze(0).to(device)
            gt_tensor = (gt * 2 - 1).unsqueeze(0).to(device)
            lp = lpips_fn(pred_tensor.to(device), gt_tensor.to(device)).item()


            rows.append([cam.image_path, psnr, ssim, lp])
            psnr_all.append(psnr)
            ssim_all.append(ssim)
            lp_all.append(lp)

            # Free memory immediately
            del pred_tensor, gt_tensor
            torch.cuda.empty_cache()

    # ---- Log aggregate numbers for this iteration -------------
    run.log({
        "iter": iter_num,
        "val/psnr": np.mean(psnr_all),
        "val/ssim": np.mean(ssim_all),
        "val/lpips": np.mean(lp_all),
    }, step=iter_num)

    # ---- Upload preview (first val view) ----------------------
    first_cam = val_cams[0].to(device)
    pred = model.forward(first_cam, active_sh_degree=model.sh_degree).clamp(0, 1).cpu().detach().numpy().transpose(1, 2,
                                                                                                                   0)
    gt = (first_cam.image.cpu().numpy().transpose(1, 2, 0))
    err = np.abs(pred - gt) / np.maximum(np.abs(pred - gt).max(), 1e-8)

    preview = np.concatenate(
        [gt, pred, np.abs(pred - gt) / np.maximum(np.abs(pred - gt).max(), 1e-8)],
        axis=1
    )  # all panels now 0‥1
    wandb_preview = wandb.Image((preview * 255).astype(np.uint8),
                                caption=f"iter {iter_num}")
    run.log({"preview": wandb_preview}, step=iter_num)

    # ---- Save model → artifact (versioned in W&B) -------------
    ply_path = Path(f"model_iter-{iter_num:04d}.ply")
    model.save_ply("./" + str(ply_path))
    art = wandb.Artifact("gaussian_cloud", type="model",
                         description="Ply dump each val interval")
    art.add_file(str(ply_path))
    run.log_artifact(art)

    # ---- Optional: frame-level table (one row per view) -------
    tbl = wandb.Table(columns=["frame", "psnr", "ssim", "lpips"], data=rows)
    run.log({f"frame_metrics/iter_{iter_num}": tbl}, step=iter_num)

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()


    print(f"✅ logged iter {iter_num} to W&B")
