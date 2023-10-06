import torch
import torchvision as tv
from detectron2.data import transforms as T

resize_op = T.ResizeShortestEdge(
    short_edge_length=800,
    max_size=1333,
)
RGB = tv.io.ImageReadMode.RGB
image_file = "./original_data/1.jpg"
mask_file = "./original_data/1.mask.png"
image = tv.io.read_image(image_file, RGB).permute(1, 2, 0)
resize = resize_op.get_transform(image)
mask = tv.io.read_image(mask_file).permute(1, 2, 0)

mask = torch.as_tensor(resize.apply_segmentation(mask.numpy())).permute(2, 0, 1) != 0

print(f"Image shape: {image.shape}")
print(f"Mask shape: {mask.shape}")
