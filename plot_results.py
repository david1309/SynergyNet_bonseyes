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

from utils.plot import plotUtils
from model_building import SynergyNet
from data.dataloader_300wlp import dataset_from_datatool

to_np = lambda tensor: tensor.detach().cpu().numpy()
to_nps = lambda tensors: [to_np(t) for t in tensors]

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

def _plot_lms(plotter, lm_3d, lm_color = (255, 150, 0)):
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

def plot_results(model, images, saving_path, targets=None, only_gt=False):
    if targets is None and only_gt:
        raise ValueError("Can't set only_gt=True without providing GT targets")
    # Get models predictions
    lms_3d, pose_para, shape_para, exp_para = model.forward_test(images.cuda(non_blocking=True))
    lms_3d, pose_para, shape_para, exp_para = to_nps([lms_3d, pose_para, shape_para, exp_para])

    # Get GT's
    plot_gt = False
    if targets is not None:
        plot_gt = True
        lms_3d_gt = to_np(targets["lm3d"])
        pose_para_gt = to_np(targets["pose_params"])

    # Plots predictions and GT
    if os.path.exists(saving_path):
        shutil.rmtree(saving_path)
    Path(saving_path).mkdir(parents=True, exist_ok=True)

    num_im = images.shape[0]
    for i in range(num_im):
        image = ToPILImage()(images[i])
        image = np.array(image)[:,:,::-1].copy()
        plotter = plotUtils(image)

        # Plot predictions
        if not only_gt:
            lm_3d = lms_3d[i].T
            head_pose = pose_para[i, :3, 0]

            lm_color = (255, 150, 0) # BGR
            _plot_lms(plotter, lm_3d, lm_color)
            ax_colors = [(0,0,255), (0,255,0), (255,0,0)] # BGR scale
            _plot_head_pose(plotter, head_pose, lm_3d, ax_colors)

        # Plot GT
        if plot_gt:
            lm_3d_gt = lms_3d_gt[i]
            head_pose_gt = pose_para_gt[i, :3]

            lm_color = (0, 150, 255) # BGR
            _plot_lms(plotter, lm_3d_gt, lm_color)
            ax_colors = [(0,0,90), (0,90,0), (90,0,0)] # BGR scale
            _plot_head_pose(plotter, head_pose_gt, lm_3d_gt, ax_colors)

        # Store plots
        output_path = os.path.join(saving_path, f"sample_{i}.jpg")
        plotter.save(output_path)


if __name__ == '__main__':
    # Config Other
    only_gt = True

    # Config Model
    args = namedtuple("args", ["arch", "img_size", "devices_id", "num_lms"])
    args.arch = "mobilenet_v2"
    args.img_size = 450
    args.devices_id = [0]
    args.num_lms = 77
    
    ckp_epoch = 10
    ckp_date = "ckpts_01h49m33s_26.03.2022"
    ckp_path = f"ckpts/{ckp_date}/model_ckpts/SynergyNet_ckp_epoch_{ckp_epoch}.pth.tar"

    # Config Data Loader
    datatool_root_dir = "/hdd1/datasets/300W_LP/output_debug/" 
    tags = ["IBUG"]
    add_transforms = []
    batch_size = 16
    workers = 4

    # Build objects and plot
    model = SynergyNet(args)
    checkpoint = torch.load(ckp_path, map_location=lambda storage, loc: storage)['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    torch.cuda.set_device(args.devices_id[0])
    model = model.cuda()

    dataset = dataset_from_datatool(datatool_root_dir, tags, add_transforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                              shuffle=False, pin_memory=True, drop_last=False)

    images, targets = next(iter(data_loader))
    saving_path = f"ckpts/{ckp_date}/images_results_test"
    plot_results(
        model, 
        images, 
        saving_path, 
        targets=targets, 
        only_gt=only_gt
        )