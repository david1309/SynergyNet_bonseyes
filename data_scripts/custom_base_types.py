"""
This Module defines the basic models for all annotations available in the
300W-LP dataset and creates their data model through python classes.
"""
from __future__ import annotations
from typing import Dict, List
import math

import numpy as np

from data.datatool_api.data_validation.base_model import BaseModel
from data.datatool_api.data_validation.attribute import Attribute
from data.datatool_api.data_validation.containers.ValList import ValList
from data.datatool_api.models.BaseTypes import BoundingBox, Headpose, Landmark2D, Landmark3D, Number


class Dataset300WLPInfo(BaseModel):
    """
    Represents metadata for the 300W-LP dataset

    Attributes:
        description: str: Dataset description
        URL: str: URL for the dataset
        version: str: Dataset version
        year: str: Year of creation
    """
    description: str = Attribute(type=str)
    url: str = Attribute(type=str)
    version: str = Attribute(type=str)
    year: str = Attribute(type=str)

    def __init__(
        self,
        description: str = None,
        url: str = None,
        version: str = None,
        year: str = None,
    ):
        args = locals().copy()
        args.pop('self')
        super().__init__(**args)

    def extract(self, inputDict: Dict) -> Dataset300WLPInfo:
        """
        Extract dataset info

        :param inputDict: Input dict containing the dataset info
        :return: Dataset300WLPInfo
        """
        self.description = inputDict['description']
        self.url = inputDict['url']
        self.version = inputDict['version']
        self.year = inputDict['year']

        return self


class Dataset300WLPLicense(BaseModel):
    """
    Represents licenses for the 300W-LP dataset

    Attributes:
        licenses: License: list of dataset's licenses
    """
    id: str = Attribute(type=str)
    description: str = Attribute(type=str)
    url: str = Attribute(type=str)
    version: str = Attribute(type=str)
    year: str = Attribute(type=str)


    def __init__(
        self,
        id: str = None,
        description: str = None,
        url: str = None,
        version: str = None,
        year: str = None
    ):
        args = locals().copy()
        args.pop('self')
        super().__init__(**args)

    def extract(self, inputDict: Dict) -> Dataset300WLPLicense:
        """
        Extract license info

        :param inputDict: Input dict containing the license info
        :return: License
        """
        self.id = inputDict['id']
        self.description = inputDict['description']
        self.url = inputDict['url']
        self.version = inputDict['version']
        self.year = inputDict['year']

        return self


class Dataset300WLPImage(BaseModel):
    """
    Represents an image metadata in the 300W-LP dataset

    Attributes:
        id: str: Image ID
        file_path: str: File path for the image
        tag: str: Name of tag / subfolder where image is located e.g. IBUG, HELEN
        width: int: Image width in pixels
        height: int: Image height in pixels
        depth: int: Number of channels in the Image
        channel_order: str: Order of R, G, B channels in the Image
    """
    id: str = Attribute(type=str)
    file_path: str = Attribute(type=str)
    tag: str = Attribute(type=str)
    width: int = Attribute(type=int)
    height: int = Attribute(type=int)
    depth: int = Attribute(type=int)
    channel_order: str = Attribute(type=str)

    def __init__(
        self, 
        id: str = None,
        file_path: str = None,
        tag: str = None,
        width: int = None,
        height: int = None,
        depth: int = None,
        channel_order: str = None
        ):
        args = locals().copy()
        args.pop('self')
        super().__init__(**args)

    def extract(self, inputDict: Dict) -> Dataset300WLPImage:
        """
        Extract the image info

        :param inputDict: Input dict containing the 300W-LP image information
        :return: CocoImage
        """
        self.id = str(inputDict['id'])
        self.file_path = str(inputDict['file_path'])
        self.tag = str(inputDict['tag'])
        self.width = inputDict['width']
        self.height = inputDict['height']
        self.depth = inputDict['depth']
        self.channel_order = str(inputDict['channel_order'])

        return self


class CustomBoundingBox2D(BoundingBox):
    """
    CustomBoundingBox2D which inherits from pre-existing type BoundingBox

    Additional Attributes:
        is_valid: bool: True if Bounding box is a valid bounding box, False otherwise
    """
    is_valid: bool = Attribute(type=bool)

    def __init__(
        self, 
        x: float = None, 
        y: float = None, 
        w: float = None, 
        h: float = None, 
        area: float = None,
        is_valid: bool = None
        ):
        super().__init__(topX=x, topY=y, w=w, h=h, area=area)
        if is_valid is not None:
            self.is_valid = is_valid

    def extract(self, inputList: List[float]) -> CustomBoundingBox2D:
        """
        Extract the bounding box from the COCO bounding box array
        :param inputList: Input array containing the bounding box values from coco whole body dataset
        :return: CustomBoundingBox2D
        """

        if len(inputList) < 4:
            raise Exception('Insufficient number of elements in the input list')

        self.topX = float(inputList[0])
        self.topY = float(inputList[1])

        if float(inputList[2]) * float(inputList[3]) > 0:
            self.w = float(inputList[2])
            self.h = float(inputList[3])
            self.area = self.w * self.h
            self.is_valid = True
        else:
            self.is_valid = False

        return self


def landmarks_to_bbox(landmarks, expand_ratio=None):
    assert isinstance(landmarks, np.ndarray) and len(landmarks.shape) == 2, 'The landmarks is not right : {}'.format(landmarks)
    assert landmarks.shape[0] == 2 or landmarks.shape[0] == 3, 'The shape of landmarks is not right : {}'.format(landmarks.shape)
    if landmarks.shape[0] == 3:
        landmarks = landmarks[:2, landmarks[-1,:].astype('bool') ]
    elif landmarks.shape[0] == 2:
        landmarks = landmarks[:2, :]
    else:
        raise Exception('The shape of landmarks is not right : {}'.format(landmarks.shape))
    assert landmarks.shape[1] >= 2, 'To get the box of landmarks, there should be at least 2 vs {}'.format(landmarks.shape)
    box = np.array([ landmarks[0,:].min(), landmarks[1,:].min(), landmarks[0,:].max(), landmarks[1,:].max() ])
    W = box[2] - box[0]
    H = box[3] - box[1]
    assert W > 0 and H > 0, 'The size of box should be greater than 0 vs {}'.format(box)
    if expand_ratio is not None:
        box[0] = int( math.floor(box[0] - W * expand_ratio) )
        box[1] = int( math.floor(box[1] - H * (expand_ratio * 3.0)) )
        box[2] = int( math.ceil(box[2] + W * expand_ratio) )
        box[3] = int( math.ceil(box[3] + H * expand_ratio) )
    return box

class Dataset300WLPAnnotation(BaseModel):
    """
    One annotation in the 300W-LP dataset

    Attributes:
    id_annotation: str: Annotation ID
    id_image: str: ID of the Image liked with the annotation
    file_path: str: path to annotation file
    tag: str: Name of tag / subfolder where image is located e.g. IBUG, HELEN
    face_landmarks_2d: ValList[Landmark2D]: List of 2D face landmarks
    count_face_landmarks_2d: int: Number of landmarks per annotation
    face_landmarks_3d: ValList[Landmark3D]: List of 3D face landmarks
    count_face_landmarks_3d: int: Number of landmarks per annotation
    face_bounding_box_2d: CustomBoundingBox2D: 2D Bounding box (ROI) of face
    vertices_path: str: Path to numpy array containing the 3D face mesh (vertices)
    head_pose: Headpose: headpose angles of face (yaw, pitch, roll)
    head_translation: ValList[Number]: Head translation in the x,y,z coordinates
    head_scale: Number: Scale factor for head
    """

    id_annotation: str = Attribute(type=str)
    id_image: str = Attribute(type=str)
    file_path: str = Attribute(type=str)
    tag: str = Attribute(type=str)
    face_landmarks_2d: ValList[Landmark2D] = Attribute(
        type=ValList,
        element_constraints=Attribute(type=Landmark2D)
        )
    count_face_landmarks_2d: int = Attribute(type=int)
    face_landmarks_3d: ValList[Landmark3D] = Attribute(
        type=ValList,
        element_constraints=Attribute(type=Landmark3D)
        )
    count_face_landmarks_3d: int = Attribute(type=int)
    face_bounding_box_2d: CustomBoundingBox2D = Attribute(type=CustomBoundingBox2D)
    vertices_path: str = Attribute(type=str)
    head_pose: Headpose = Attribute(type=Headpose)
    head_translation: List[Number] = Attribute(type=list)
    head_scale : Number= Attribute(type=Number)
    shape_params: List[Number] = Attribute(type=list)
    exp_params: List[Number] = Attribute(type=list)

    def __init__(
        self,
        id_annotation: str = None,
        id_image: str = None,
        file_path: str = None,
        tag: str = None,
        face_landmarks_2d: ValList[Landmark2D] = None,
        count_face_landmarks_2d: int = None,
        face_landmarks_3d: ValList[Landmark3D] = None,
        count_face_landmarks_3d: int = None,
        face_bounding_box_2d: CustomBoundingBox2D = None,
        vertices_path: str = None,
        head_pose: Headpose = None,
        head_translation: List[Number] = None,
        head_scale: Number = None,
        shape_params : List[Number] = None,
        exp_params : List[Number] = None
        ):
        args = locals().copy()
        args.pop('self')
        super().__init__(**args)

    @staticmethod
    def extract_2d_landmarks(lm_2d_array: np.array) -> List:
        num_lm = lm_2d_array.shape[1]
        lm_list = ValList()

        for idx in range(num_lm):
            landmark = Landmark2D(
                index=idx,
                x=lm_2d_array[0, idx],
                y=lm_2d_array[1, idx]
                )

            lm_list.append(landmark)

        return lm_list

    @staticmethod
    def extract_3d_landmarks(lm_3d_array: np.array) -> List:
        lm_3d_array = lm_3d_array.astype(np.float64)
        num_lm = lm_3d_array.shape[1]
        lm_list = ValList()

        for idx in range(num_lm):
            landmark = Landmark3D(
                index=idx,
                x=lm_3d_array[0, idx],
                y=lm_3d_array[1, idx],
                z=lm_3d_array[2, idx]
                )

            lm_list.append(landmark)

        return lm_list

    def extract(self, inputDict: Dict) -> Dataset300WLPAnnotation:
        """
        Extract the annotation data from the 300W-LP annotation object

        :param inputDict: Input dict containing one 300W-LP face annotation
        :return: Dataset300WLPAnnotation
        """
        # General info
        self.id_annotation = str(inputDict['id_annotation'])
        self.id_image = str(inputDict['id_image'])
        self.file_path = str(inputDict['file_path'])
        self.tag = str(inputDict['tag'])

        # 3d landmarks
        self.face_landmarks_3d = self.extract_3d_landmarks(inputDict['pt3d'])
        self.count_face_landmarks_3d = len(self.face_landmarks_3d)

        # 2d landmarks
        self.face_landmarks_2d = self.extract_2d_landmarks(inputDict['pt2d'])
        self.count_face_landmarks_2d = len(self.face_landmarks_2d)

        # Bounding box (computed from 3D landmarks)
        expand_ratio = 0.22
        face_bb2d = landmarks_to_bbox(inputDict['pt3d'], expand_ratio=expand_ratio)
        face_bb2d[2] = face_bb2d[2] - face_bb2d[0]  # transform width to be w.r.t. topX
        face_bb2d[3] = face_bb2d[3] - face_bb2d[1]  # transform height to be w.r.t. topY
        self.face_bounding_box_2d = CustomBoundingBox2D().extract(face_bb2d)

        # Vertices (3D face mesh) file path
        self.vertices_path = str(inputDict['vertices_path'])

        # Morphable model parameters
        ## Pose:
        pose_para = inputDict['Pose_Para'].T.astype(np.float64)
        s = pose_para[-1, 0]
        angles = pose_para[:3, 0]
        t = pose_para[3:6, 0]
        self.head_scale = s
        self.head_translation = list(t)
        self.head_pose = Headpose(yaw= angles[2]*90/np.pi, pitch=angles[1]*90/np.pi, roll= angles[0]*90/np.pi, 
                                 confidence=1.0)

        ## Shape & Expression:
        self.shape_params = list(np.float64(inputDict["Shape_Para"][:, 0]))
        self.exp_params = list(np.float64(inputDict["Exp_Para"][:, 0]))

        return self
