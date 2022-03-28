"""
This is an example data loader for pytorch which uses the datatool API to load
the datatool output JSON and read samples.

The task is the regression of 68 3D facial landmarks, and the regresion of the
subjects head pose (yaw, pitch roll).

Input:
The input is a datatool output directory containing the dataset.json and samples directory

Assumptions are:
    - The datatool output contains annotations for:
        - 68 3D facial landmarks
        - head pose (yaw, pitch roll)

    - The samples are RGB images
    - The models needs 450 x 450 size input tensor which contains face image
"""
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms, Normalize
from PIL import Image
import os
from pathlib import Path

from data.datatool_api.config.APIConfig import DTAPIConfig
from data.custom_dataset_model import DTDatasetCustom

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_sample(img, face_lm_2d, bbox, saving_path=None):
    if isinstance(img, torch.Tensor):
        img_arr = np.transpose(np.array(img), (1, 2, 0))
    else:
        img_arr = np.array(img)

    # Plot image
    fig, ax = plt.subplots(1,1)
    plt.imshow(img_arr)

    # Plot landmarks
    plt.scatter(face_lm_2d[0,:],face_lm_2d[1,:], s=3)

    # Plot bbox
    # rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=2, edgecolor='orange',facecolor='none')
    rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],linewidth=2, edgecolor='orange',facecolor='none')
    ax.add_patch(rect)
    if saving_path:
        ax.get_figure().savefig(saving_path)

class PTDataset300WLP(Dataset):
    def __init__(
        self,
        data_root_dir: str,
        tags: list,
        add_transforms: list = [],
        operating_mode: str = 'memory',
        **kwargs
        ):
        """
        Instantiate the dataset instance

        :param data_root_dir: Root Directory containing all the tags produced
        by the datatool

        :param tags: Name of each subfolder (tag), each one containing the
        "sample_files" directory and "dataset.json" file

        :param operating_mode: Operating mode for the datatool api to handle
        the dataset, based on the data size, user can choose
        [memory, disk or ramdisk]

        :param kwargs: Additional keyword arguments
        """
        Dataset.__init__(self)
        self.data_root_dir = data_root_dir
        self.tags = tags
        self.kwargs = kwargs

        # No need to validate when loading back the API output
        DTAPIConfig.disable_validation()

        # Load the dataset.json for each tag
        self.annotations = {}
        self.usable_annotations = []
        for tag in tags:
            _dataset = DTDatasetCustom(
                name=f'300w_lp_dataset_{tag}',
                operatingMode=operating_mode
                )
            dataset = _dataset.load_from_json(
                os.path.join(self.data_root_dir, tag, 'dataset.json'),
                element_list=['annotations']
                )

            # Filtered list validating if annotations requirements are met
            for annot_id, annotation in dataset.annotations.items():
                flags = []
                flags.append(
                    annotation.count_face_landmarks_2d >= self.kwargs['min_landmark_count']
                    )
                flags.append(
                    annotation.count_face_landmarks_3d >= self.kwargs['min_landmark_count']
                )
                flags.append(
                    len(annotation.shape_params) == self.kwargs['shape_param_size']
                )
                flags.append(
                    len(annotation.exp_params) == self.kwargs['expression_param_size']
                )

                if all(flags):
                    self.usable_annotations.append(annot_id)
                    self.annotations[annot_id] = annotation

        # Set of transforms
        base_transforms = [
                transforms.Resize(self.kwargs['model_input_size']),
                transforms.ToTensor()
            ]
        all_transforms = [*base_transforms, *add_transforms]
        self.transform = transforms.Compose(all_transforms)

    def __len__(self):
        return len(self.usable_annotations)

    def __getitem__(self, index):
        annot_id = self.usable_annotations[index]
        annotation = self.annotations.get(annot_id)
        model_input_size = self.kwargs['model_input_size']

        # Landmarks
        lm3d = annotation.face_landmarks_3d
        bb2d = annotation.face_bounding_box_2d

        # Bounding Bounding box
        bbox = torch.Tensor([
            bb2d.topX,
            bb2d.topY,
            bb2d.w,
            bb2d.h
        ])

        # Morphable parameters (normalized by 1e7, 10, image height, and 100)
        shape_params = torch.Tensor(annotation.shape_params) / 1e7
        exp_params = torch.Tensor(annotation.exp_params) / 10

        # BFM documentation states that:
        # angles: [3,]. x, y, z angles
        # x: pitch.
        # y: yaw.
        # z: roll.
        head_pose_ = annotation.head_pose
        deg2rad = (np.pi / 180)
        head_pose = [
            head_pose_.pitch * deg2rad,
            head_pose_.yaw * deg2rad,
            head_pose_.roll * deg2rad,
        ]
        head_translation = annotation.head_translation
        head_translation = [t / model_input_size[0] for t in head_translation]
        head_scale = [annotation.head_scale * 100]
        pose_params = torch.Tensor([head_pose + head_translation + head_scale]).squeeze()

        # Obtain Tensor for 3D landmarks
        lm3d_temp = np.empty(shape=(3, annotation.count_face_landmarks_3d))
        for i, lm in enumerate(lm3d):
            lm3d_temp[0,i] = lm.x #(lm.x - bb2d.topX) * model_input_size[0] / bb2d.w
            lm3d_temp[1,i] = lm.y #(lm.y - bb2d.topY) * model_input_size[1] / bb2d.h
            lm3d_temp[2,i] = lm.z
        lm3d = torch.from_numpy(lm3d_temp)

        # Read image for the sample and Apply transforms
        image_name = annotation.id_image + '.jpg'
        tag = annotation.tag
        img = Image.open(os.path.join(self.data_root_dir, tag, 'sample_files', image_name))
        img = self.transform(img)

        target = {
            "lm3d" : lm3d,
            "pose_params" : pose_params,
            "shape_params" : shape_params,
            "exp_params" : exp_params,
            "bbox" : bbox
        }
        return img, target


def dataset_from_datatool(datatool_root_dir, tags, add_transforms=[]):
    params = {
        'model_input_size': (450, 450),  # Width x Height of input tensor for the model
        'min_landmark_count': 68,  # Min number of landmarks
        'shape_param_size': 199,
        'expression_param_size': 29
    }
    dataset = PTDataset300WLP(
        data_root_dir=datatool_root_dir,
        tags=tags,
        operating_mode='memory',
        add_transforms=add_transforms,
        **params
        )

    return dataset

# def main():
#     # Input arguments
#     params = {
#         'model_input_size': (450, 450),  # Width x Height of input tensor for the model
#         'min_landmark_count': 68,  # Min number of landmarks
#         'shape_param_size': 199,
#         'expression_param_size': 29
#     }

#     data_root_dir = '/hdd1/datasets/300W_LP/output_debug/'
#     tags = ["IBUG", "IBUG_Flip"] #, "AFW", "AFW_Flip"]

#     # Create dataset instance
#     normalize = Normalize(
#         mean=[0.498, 0.498, 0.498],
#         std=[0.229, 0.229, 0.229]
#         )
#     add_transforms = [normalize]


#     # Build and test Data Loader
#     dataset = PTDataset300WLP(
#         data_root_dir=data_root_dir,
#         tags=tags,
#         operating_mode='memory',
#         add_transforms=add_transforms,
#         **params
#         )

#     for i in range(len(dataset)):
#         img, target = dataset.__getitem__(i)
