import supervisely as sly
import json
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv(os.path.expanduser("supervisely.env"))
api = sly.Api()
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(29031))
image_files = sly.fs.list_files_recursively("./project/dataset_0/img/")
for image_file in image_files:
    image_name = os.path.basename(image_file)
    ann_file = "./project/dataset_0/ann/" + image_name + ".json"
    with open(ann_file) as f:
        ann_json = json.load(f)
        ann = sly.Annotation.from_json(ann_json, project_meta)
        image = sly.image.read(image_file)
        mask = np.zeros(image.shape)
        ann.draw_pretty(
            mask,
            output_path="./masks/pexels_162223.png",
            thickness=7,
        )
        sly.image.write("./input_data/masks/" + image_name, mask)