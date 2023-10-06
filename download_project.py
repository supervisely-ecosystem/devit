import supervisely as sly
from dotenv import load_dotenv
import os

load_dotenv(os.path.expanduser("supervisely.env"))
api = sly.Api()

sly.download(
            api=api,
            project_id=29031,
            dest_dir="./project/",
            dataset_ids=[77072, 77073, 77074, 77075],
        )