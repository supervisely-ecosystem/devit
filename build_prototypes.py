import torch
import os
import torchvision as tv
from detectron2.data import transforms as T
from torchvision.transforms import functional as tvF

torch.set_grad_enabled(False)
to_pil = tv.transforms.functional.to_pil_image
from collections import defaultdict
from tqdm import tqdm
import torchvision.ops as ops
import torch.nn.functional as F
import supervisely as sly

RGB = tv.io.ImageReadMode.RGB

pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.120, 57.375]).view(3, 1, 1)
normalize_image = lambda x: (x - pixel_mean) / pixel_std


def iround(x):
    return int(round(x))


def resize_to_closest_14x(img):
    h, w = img.shape[1:]
    h, w = max(iround(h / 14), 1) * 14, max(iround(w / 14), 1) * 14
    return tvF.resize(img, (h, w), interpolation=tvF.InterpolationMode.BICUBIC)


resize_op = T.ResizeShortestEdge(
    short_edge_length=800,
    max_size=1333,
)

# reading metas
class2images = {"hummingbird": []}
image_files = sly.fs.list_files_recursively("./input_data/img/")
for image_file in image_files:
    image_name = os.path.basename(image_file)
    mask_file = "./input_data/masks/" + image_name[:-4] + "png"
    class2images["hummingbird"].append((image_file, mask_file))

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
device = 0
model = model.to(device)
class2tokens = {}
for cls, images in tqdm(class2images.items()):
    class2tokens[cls] = []
    for image_file, mask_file in images:
        image = tv.io.read_image(image_file, RGB).permute(1, 2, 0)
        resize = resize_op.get_transform(image)
        mask = tv.io.read_image(mask_file).permute(1, 2, 0)

        mask = torch.as_tensor(resize.apply_segmentation(mask.numpy())).permute(2, 0, 1) != 0
        mask = mask[0, :, :].unsqueeze(0)
        image = torch.as_tensor(resize.apply_image(image.numpy())).permute(2, 0, 1)

        image14 = resize_to_closest_14x(image)
        mask_h, mask_w = image14.shape[1] // 14, image14.shape[2] // 14
        nimage14 = normalize_image(image14)[None, ...]
        r = model.get_intermediate_layers(
            nimage14.to(device), return_class_token=True, reshape=True
        )
        patch_tokens = r[0][0][0].cpu()
        mask14 = tvF.resize(mask, (mask_h, mask_w))
        if mask14.sum() <= 0.5:
            continue
        avg_patch_token = (mask14 * patch_tokens).flatten(1).sum(1) / mask14.sum()
        class2tokens[cls].append(avg_patch_token)

for cls in class2tokens:
    class2tokens[cls] = torch.stack(class2tokens[cls]).mean(dim=0)

classes = ["hummingbird"]
prototypes = F.normalize(torch.stack([class2tokens[c] for c in classes]), dim=1)
category_dict = {"prototypes": prototypes, "label_names": classes}
torch.save(category_dict, "hummingbird_prototypes.pth")
