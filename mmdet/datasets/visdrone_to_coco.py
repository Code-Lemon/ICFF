import os
from pathlib import Path
import cv2
import fire
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import save_json
from tqdm import tqdm

CATEGORY_ID_TO_NAME = {
    "0": "ignored regions",
    "1": "pedestrian",
    "2": "people",
    "3": "bicycle",
    "4": "car",
    "5": "van",
    "6": "truck",
    "7": "tricycle",
    "8": "awning-tricycle",
    "9": "bus",
    "10": "motor",
    "11": "others",
}

CATEGORY_ID_REMAPPING = {
    "1": "0",
    "2": "1",
    "3": "2",
    "4": "3",
    "5": "4",
    "6": "5",
    "7": "6",
    "8": "7",
    "9": "8",
    "10": "9",

}

NAME_TO_COCO_CATEGORY = {
    "pedestrian": {"name": "pedestrian", "supercategory": "person"},
    "people": {"name": "people", "supercategory": "person"},
    "bicycle": {"name": "bicycle", "supercategory": "bicycle"},
    "car": {"name": "car", "supercategory": "car"},
    "van": {"name": "van", "supercategory": "truck"},
    "truck": {"name": "truck", "supercategory": "truck"},
    "tricycle": {"name": "tricycle", "supercategory": "motor"},
    "awning-tricycle": {"name": "awning-tricycle", "supercategory": "motor"},
    "bus": {"name": "bus", "supercategory": "bus"},
    "motor": {"name": "motor", "supercategory": "motor"},
}


# def visdrone_to_coco_bbox(visdrone_bbox, img_width, img_height):
#     """
#     将 VisDrone 格式的 bbox 转为 COCO 格式。
#
#     参数：
#         visdrone_bbox: [cx, cy, w, h]，归一化坐标（值在0~1之间）
#         img_width: 图像宽度（像素）
#         img_height: 图像高度（像素）
#
#     返回：
#         coco_bbox: [x, y, w, h]，像素坐标
#     """
#     cx, cy, w_norm, h_norm = visdrone_bbox
#     w = round(float(w_norm) * img_width, 1)
#     h = round(float(h_norm) * img_height, 1)
#     x = round((float(cx) * img_width) - w / 2, 1)
#     y = round((float(cy) * img_height) - h / 2, 1)
#     return [x, y, w, h]

def visdrone_to_coco(
    data_folder_dir,
    output_file_path,
    input_ann_folder,
    category_id_remapping=None,
):
    """
    Converts visdrone-det annotations into coco annotation.

    Args:
        data_folder_dir: str
            'VisDrone2019-DET-val' folder directory
        output_file_path: str
            Output file path
        category_id_remapping: dict
            Used for selecting desired category ids and mapping them.
            If not provided, VisDrone2019-DET mapping will be used.
            format: str(id) to str(id)
    """
    a =[]
    # init paths/folders
    input_image_folder = data_folder_dir
    input_ann_folder = input_ann_folder

    image_filepath_list = os.listdir(input_image_folder)

    Path(output_file_path).parents[0].mkdir(parents=True, exist_ok=True)

    if category_id_remapping is None:
        category_id_remapping = CATEGORY_ID_REMAPPING

    # init coco object
    coco = Coco()
    # append categories
    for category_id, category_name in CATEGORY_ID_TO_NAME.items():
        if category_id in category_id_remapping.keys():
            remapped_category_id = category_id_remapping[category_id]
            coco_category = NAME_TO_COCO_CATEGORY[category_name]
            coco.add_category(
                CocoCategory(
                    id=int(remapped_category_id),
                    name=coco_category["name"],
                    supercategory=coco_category["supercategory"],
                )
            )

    # convert visdrone annotations to coco
    # for image_filename in image_filepath_list:
    for image_filename in tqdm(image_filepath_list):
        # get image properties
        image_filepath = str(Path(input_image_folder) / image_filename)
        annotation_filename = image_filename.split(".jpg")[0] + ".txt"
        annotation_filepath = str(Path(input_ann_folder) / annotation_filename)
        image = Image.open(image_filepath)
        cocoimage_filename = str(Path(image_filepath)).split(str(Path(data_folder_dir)))[1]
        if cocoimage_filename[0] == os.sep:
            cocoimage_filename = cocoimage_filename[1:]
        # create coco image object
        height = image.size[1]
        width = image.size[0]
        coco_image = CocoImage(file_name=cocoimage_filename, height=image.size[1], width=image.size[0])
        # parse annotation file
        file = open(annotation_filepath, "r")
        lines = file.readlines()
        for line in lines:
            a.append(line[0])
            # parse annotation bboxes
            new_line = line.strip("\n").split(",")
            bbox = [int(item) for item in new_line[:4]]
            # parse category id and name
            category_id = new_line[5]
            if category_id in category_id_remapping.keys():
                category_name = CATEGORY_ID_TO_NAME[category_id]
                remapped_category_id = category_id_remapping[category_id]
            else:
                continue
            # create coco annotation and append it to coco image
            coco_annotation = CocoAnnotation.from_coco_bbox(
                bbox=bbox,
                category_id=int(remapped_category_id),
                category_name=category_name,
            )
            if coco_annotation.area > 0:
                coco_image.add_annotation(coco_annotation)
        coco.add_image(coco_image)

    save_path = output_file_path
    save_json(data=coco.json, save_path=save_path)


if __name__ == "__main__":
    # fire.Fire(visdrone_to_coco)
    data_folder_dir = "/mnt/sdb/lx/data/VisDrone/test/"
    input_ann_folder = "/mnt/sdb/lx/data/VisDrone/labels/test/"
    output_file_path = "/mnt/sdb/lx/data/VisDrone/annotations/instances_test.json"
    visdrone_to_coco(data_folder_dir, output_file_path, input_ann_folder, category_id_remapping=None)