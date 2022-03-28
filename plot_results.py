from collections import namedtuple
from pathlib import Path
import os
import shutil

import numpy as np
import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from math import cos, sin, sqrt
import cv2
import matplotlib.pyplot as plt

from utils.plot import plotUtils
from model_building import SynergyNet
from data.dataloader_300wlp import dataset_from_datatool
import time

to_np = lambda tensor: tensor.detach().cpu().numpy()
to_nps = lambda tensors: [to_np(t) for t in tensors]

def draw_landmarks(img, pts, color, saving_path):
    height, width = img.shape[:2]
    base = 6.4 
    plt.figure(figsize=(base, height / width * base))
    plt.imshow(img[:, :, ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    markeredgecolor = color

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        alpha = 0.8
        markersize = 3.5
        lw = 1.5

        # close eyes and mouths
        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
        plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                color=color, lw=lw, alpha=alpha - 0.1)
        plot_close(41, 36)
        plot_close(47, 42)
        plot_close(59, 48)
        plot_close(67, 60)

        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

            plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                        color=color,
                        markeredgecolor=markeredgecolor, alpha=alpha)
    plt.tight_layout(pad=0)
    plt.savefig(saving_path, dpi=200)
    time.sleep(1.25)
    image = cv2.imread(saving_path)
    image = cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    os.remove(saving_path)
    return image

def _draw_3d_axis(
    img,
    yaw,
    pitch,
    roll,
    tdx=None,
    tdy=None,
    lm=None,
    ax_colors=None,
    size=100,
    ):
    if (tdx is None) or (tdy is None):
        tdx = lm[0,30]
        tdy = lm[1,30]

    minx, maxx = np.min(lm[0, :]), np.max(lm[0, :])
    miny, maxy = np.min(lm[1, :]), np.max(lm[1, :])
    llength = sqrt((maxx - minx) * (maxy - miny))
    size = llength * 0.5

    # X-Axis pointing to right. drawn in red (Pitch)
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green (Yaw)
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue (Roll)
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    minus=0

    # Plot
    if ax_colors is None:
        # BGR scale
        ax_colors = [(0,0,255), (0,255,0), (255,0,0)]
    cv2.line(img, (int(tdx), int(tdy)-minus), (int(x1),int(y1)), ax_colors[0],4)
    cv2.line(img, (int(tdx), int(tdy)-minus), (int(x2),int(y2)), ax_colors[1],4)
    cv2.line(img, (int(tdx), int(tdy)-minus), (int(x3),int(y3)), ax_colors[2],4)

    return img

def _plot_lms(plotter, lm_3d, lm_color=(255, 150, 0), lm_with_lines=False):
    if lm_with_lines:
        lm_color = (lm_color[2]/255, lm_color[1]/255, lm_color[0]/255) # BGR to RGB
        plotter.image = draw_landmarks(
                            plotter.image, 
                            lm_3d, 
                            lm_color,
                            "tmp.jpg"
                            )
    else:
        lms = []
        num_lm = lm_3d.shape[1]
        for i in range(num_lm):
            lms.append({
                'idx': i,
                'x': lm_3d[0, i],
                'y': lm_3d[1, i]
                })

        plotter.plot_landmarks(
            landmarks=lms,
            color=lm_color,
            radius=2,
            index_font_scale=0.0,
            index_offset=(1, -1)
            )

def _plot_head_pose(plotter, head_pose, lm_3d, ax_colors=None):
    pitch = head_pose[0] # red
    yaw = -1 * head_pose[1]  # green
    roll = head_pose[2]  # blue

    plotter.image = _draw_3d_axis(
        plotter.image,
        yaw,
        pitch,
        roll,
        lm=lm_3d,
        ax_colors=ax_colors
    )

def plot_results(
    model, 
    images, 
    saving_path, 
    lm_with_lines=False,
    targets=None, 
    only_gt=False
    ):
    if targets is None and only_gt:
        raise ValueError("Can't set only_gt=True without providing GT targets")

    # Get GT's
    plot_gt = False
    bbox = None
    if targets is not None:
        plot_gt = True
        lms_3d_gt = to_np(targets["lm3d"])
        pose_para_gt = to_np(targets["pose_params"])
        bbox = targets["bbox"]

    # Get models predictions
    if not only_gt:
        model.eval()
        with torch.no_grad():
            lms_3d, pose_para = model.forward_test(images, bbox)
            lms_3d, pose_para = to_nps([lms_3d, pose_para])

    # Plots predictions and GT
    if saving_path is not None:
        if os.path.exists(saving_path):
            shutil.rmtree(saving_path)
        Path(saving_path).mkdir(parents=True, exist_ok=True)

    num_im = images.shape[0]
    images_plotted = np.zeros_like(images)
    for i in range(num_im):
        image = ToPILImage()(images[i])
        image = np.array(image)[:,:,::-1].copy()
        plotter = plotUtils(image)

        # Plot predictions
        if not only_gt:
            lm_3d = lms_3d[i].T
            head_pose = pose_para[i, :3, 0]

            lm_color = (255, 150, 0) # BGR
            _plot_lms(plotter, lm_3d, lm_color, lm_with_lines=lm_with_lines)
            ax_colors = [(0,0,255), (0,255,0), (255,0,0)] # BGR scale
            _plot_head_pose(plotter, head_pose, lm_3d, ax_colors)

        # Plot GT
        if plot_gt:
            lm_3d_gt = lms_3d_gt[i]
            head_pose_gt = pose_para_gt[i, :3]

            lm_color = (0, 150, 255) # BGR
            _plot_lms(plotter, lm_3d_gt, lm_color, lm_with_lines=lm_with_lines)
            ax_colors = [(0,0,90), (0,90,0), (90,0,0)] # BGR scale
            _plot_head_pose(plotter, head_pose_gt, lm_3d_gt, ax_colors)

        # Store images and save plots
        images_plotted[i] = plotter.image.T

        if saving_path is not None:
            output_path = os.path.join(saving_path, f"sample_{i}.jpg")
            plotter.save(output_path)
            
    images_plotted = np.transpose(images_plotted, (0, 3, 2, 1)) # (batch, h, w, c)
    return images_plotted
            


if __name__ == '__main__':
    # Config General
    args = namedtuple("args", ["use_cuda", "arch", "img_size", "num_lms", "crop_images", "use_rot_inv", "bfm_path"])
    only_gt = False
    args.use_rot_inv = False
    args.use_cuda = False
    args.crop_images = False
    lm_with_lines = True

    seed = 13
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Config Model
    args.arch = "mobilenet_v2"
    args.img_size = 450
    args.num_lms = 77
    args.bfm_path = "bfm_utils/morphable_models/BFM.mat"

    ckp_epoch = 38
    ckp_name = "loss_weight_100_all_28.03.2022_14h23m36s"
    ckp_path = f"ckpts/{ckp_name}/model_ckpts/SynergyNet_ckp_epoch_{ckp_epoch}.pth.tar"

    # Config Data Loader
    datatool_root_dir = "/root/300wlp/"
    tags = ["IBUG"]
    add_transforms = []
    batch_size = 16
    workers = 4

    # Build objects
    if only_gt:
        model = None
        pin_memory = False
    else:
        print(f">>> Loading model from '{ckp_path}' ...")
        device = torch.device(f"cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
        args.device = device
        pin_memory = (args.device.type == "gpu")
        model = SynergyNet(args)
        checkpoint = torch.load(ckp_path, map_location=lambda storage, loc: storage)['state_dict']
        model.load_state_dict(checkpoint, strict=False)

    print(f">>> Loading data tool from '{datatool_root_dir}' with tags {tags} ...")
    dataset = dataset_from_datatool(datatool_root_dir, tags, add_transforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                              shuffle=True, pin_memory=pin_memory, drop_last=False)

    # Plot images
    images, targets = next(iter(data_loader))
    saving_path = f"ckpts/{ckp_name}/images_results_test"
    print(f">>> Plotting images ...")
    plot_results(
        model,
        images,
        saving_path,
        lm_with_lines=lm_with_lines,
        targets=targets,
        only_gt=only_gt
        )
    print(f">>> Images stored in '{saving_path}")