from __future__ import annotations
from data.datatool_api.models.base.BaseDTDataset import BaseDTDataset
from data.datatool_api.data_validation.containers.StorageValDict import StorageValDict
from data.datatool_api.data_validation.containers.ValDict import ValDict
from .custom_base_types import *
from typing import Tuple
import pandas


class DTDatasetCustom(BaseDTDataset):
    """
    Attributes:
        info: Dataset300WLPInfo: Dataset Info
        licenses: ValDict: All licenses for the dataset with their respective categories
        images: StorageValDict: Image info for all images in the dataset
        annotations: StorageValDict: Annotations for all images in the dataset
    """
    info: Dataset300WLPInfo = Attribute(type=Dataset300WLPInfo)

    licenses: ValDict[str, Dataset300WLPLicense] = Attribute(type=ValDict,
                                                           element_constraints=
                                                           Attribute(type=Dataset300WLPLicense))

    images: StorageValDict[str, Dataset300WLPImage] = Attribute(type=StorageValDict,
                                                       element_constraints=Attribute(type=Dataset300WLPImage))


    annotations: StorageValDict[str, Dataset300WLPAnnotation] = Attribute(type=StorageValDict,
                                                                 element_constraints=
                                                                 Attribute(type=Dataset300WLPAnnotation))
    def __init__(self, name: str, operatingMode: str):
        super().__init__(name, operatingMode)

    def __del__(self):
        super().__del__()

    """
    Needs to be implemented by the datatool creator
    """

    def to_pandas_frame(self, keep_original: bool = False, column_subset: List = None) -> List[Tuple[str,
                                                                                                     pandas.DataFrame]]:
        """
        Pandas dataframe generator for dataset

        :param keep_original: If original dataset needs to be kept in memory, if False, the original dataset object
        can be modified, by popping samples from it.
        :param column_subset: If only a subset of all columns are needed in the dataframe
        :return: List of (dataframe_name, pandas.Dataframe)
        """
        col_subset = {}
        if column_subset is not None and type(column_subset) == list:
            for s in column_subset:
                col_subset[s] = True
        output_dict = {}

        annotation_keys = list(self.annotations.keys())
        count = 0
        for key in annotation_keys:
            if keep_original is False:
                annotation = self.annotations.pop(key)
            else:
                annotation = self.annotations.get(key)
            flattened_annotation = {}
            annotation.flatten(flattened_annotation, 'annotation')

            # Export everything in flattened dict
            for k in output_dict:
                output_dict[k].append(flattened_annotation.pop(k, None))

            for k, v in flattened_annotation.items():
                if len(col_subset) == 0 or k in column_subset:
                    try:
                        output_dict[k].append(v)
                    except Exception as e:
                        output_dict[k] = [None] * count
                        output_dict[k].append(v)
            count += 1
        return [('annotations', pandas.DataFrame(output_dict))]

    def to_pandas_frame_for_report(self) -> List[Tuple[str, pandas.DataFrame, List[str]]]:
        """
        Get the pandas dataframe which is used for report generation. This method changes the column names to the
        expected names for columns in the report and only exports the columns which are required in the report.

        In addition it also export a column name list along with each dataframe holding the columns names which should
        be used to generate the interaction plots in the report.

        :return: List of (dataframe_name, pandas.Dataframe, List[columns_needed_for_interaction_plots])
        """
        # Columns to export in the data frame for reporting
        report_columns = {
            'annotation.id_image': 'Image Id',
            'annotation.count_face_landmarks_2d': 'Face Landmarks Count',
            'annotation.face_bounding_box_2d.area': 'Face BB2D Area',
            'annotation.face_bounding_box_2d.is_valid': 'Face BB2D Validity',
            'annotation.head_pose.yaw': 'Head pose Yaw',
            'annotation.head_pose.pitch': 'Head pose Pitch',
            'annotation.head_pose.roll': 'Head pose Roll',
            'annotation.head_translation': 'Head Translation',
            'annotation.head_scale': 'Head Scale',
        }

        report_interaction_columns = [
            'Face BB2D Area',
            'Head pose Yaw',
            'Head pose Pitch',
            'Head pose Roll',
            'Head Translation',
            'Head Scale',
        ]

        frame_out = self.to_pandas_frame(keep_original=False, column_subset=list(report_columns.keys()))
        out = []
        for tup in frame_out:
            tup[1].rename(columns=report_columns, inplace=True)
            out.append((tup[0], tup[1], report_interaction_columns))
        return out
