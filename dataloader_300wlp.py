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
import sys
import inspect


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
        data_dir: str, 
        add_transforms: list = [],
        operating_mode: str = 'memory', 
        **kwargs
        ):
        """
        Instantiate the dataset instance

        :param data_dir: Directory containing the datatool output which contains
        the "sample_files" directory and "dataset.json" file
        :param operating_mode: Operating mode for the datatool api to handle
        the dataset, based on the data size, user can choose
        [memory, disk or ramdisk]
        :param kwargs: Additional keyword arguments
        """
        Dataset.__init__(self)
        self.data_dir = data_dir
        self.operating_mode = operating_mode
        self.kwargs = kwargs

        # No need to validate when loading back the API output
        DTAPIConfig.disable_validation()

        # Load the dataset.json using the datatool API
        _dataset = DTDatasetCustom(
            name='300w_lp_dataset',
            operatingMode=self.operating_mode
            )
        self.dataset = _dataset.load_from_json(
            os.path.join(self.data_dir,'dataset.json'),
            element_list=['images', 'annotations']
            )

        # Diltered list validating if annotations requirements are met
        self.usable_annotations = []
        for annot_id, annotation in self.dataset.annotations.items():
            flags = []
            flags.append(
                annotation.count_face_landmarks_2d == self.kwargs['min_landmark_count']
                )
            flags.append(
                annotation.count_face_landmarks_3d == self.kwargs['min_landmark_count']
            )
            flags.append(
                len(annotation.shape_params) == self.kwargs['shape_param_size']
            )
            flags.append(
                len(annotation.exp_params) == self.kwargs['expression_param_size']
            )

            if all(flags):
                self.usable_annotations.append(annot_id)

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
        # Get the annotation for the index
        annot_id = self.usable_annotations[index]
        annotation = self.dataset.annotations.get(annot_id)
        lm3d = annotation.face_landmarks_3d
        bb2d = annotation.face_bounding_box_2d
        shape_params = np.array(annotation.shape_params)
        exp_params = np.array(annotation.exp_params)
        head_pose_ = annotation.head_pose
        head_pose = np.array([head_pose_.yaw, head_pose_.pitch, head_pose_.roll])
        model_input_size = self.kwargs['model_input_size']

        # Adjust morphable model params


        # Adjust landmark values for new coordinate system of cropped image
        lm3d_temp = np.empty(shape=(3, annotation.count_face_landmarks_3d))
        for i, lm in enumerate(lm3d):
            lm3d_temp[0,i] = (lm.x - bb2d.topX) * model_input_size[0] / bb2d.w
            lm3d_temp[1,i] = (lm.y - bb2d.topY) * model_input_size[1] / bb2d.h
            lm3d_temp[2,i] = lm.z
        lm3d = lm3d_temp

        # Read image for the sample and Apply transforms
        image_name = annotation.id_image + '.jpg'
        img = Image.open(os.path.join(self.data_dir, 'sample_files', image_name))

        bbox = [
            bb2d.topX, 
            bb2d.topY,
            bb2d.w,
            bb2d.h
        ]

        # base_path = "example_dataloaders/sample_imgs"
        # Path(base_path).mkdir(parents=True, exist_ok=True)
        # saving_path =  annotation.id_image + "_ori" + '.jpg'
        # plot_sample(img, lm3d, bbox, saving_path=os.path.join(base_path, saving_path))
        # img_ = np.array(img)     
        # print((img_.max(), img_.min(), img_.mean(), img_.std()))
        
        img = img.crop((
            bb2d.topX, 
            bb2d.topY,
            bb2d.topX + bb2d.w,
            bb2d.topY + bb2d.h
            ))       

        img = self.transform(img)
        # print((img.max(), img.min(), img.mean(), img.std()))
        # saving_path =  annotation.id_image + "_trans" + '.jpg'
        # plot_sample(img, lm3d, bbox, saving_path=os.path.join(base_path, saving_path))

        target = {
            "lm3d" : lm3d,
            "shape_params" : shape_params,
            "exp_params" : exp_params,
            "head_pose" : head_pose,
        }
        return img, target



def main():
    params = {
        'model_input_size': (450, 450),  # Width x Height of input tensor for the model
        'min_landmark_count': 68,  # Min number of landmarks 
        'shape_param_size': 199,
        'expression_param_size': 29
    }

    data_dir = '/hdd1/datasets/300W_LP/output_debug/IBUG'

    # Create dataset instance
    normalize = Normalize(
        mean=[0.498, 0.498, 0.498], 
        std=[0.229, 0.229, 0.229]
        )
    add_transforms = [normalize]

    dataset = PTDataset300WLP(
        data_dir=data_dir, 
        operating_mode='memory', 
        add_transforms=add_transforms,
        **params
        )

    # Test it
    for i in range(len(dataset)):
        img, target = dataset.__getitem__(i)


if __name__ == main():
    main()
