import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from constants import IMAGE_SIZE
from utils import pad_to_desired_size


def image_to_tensor(image):
    transform = transforms.Compose([transforms.PILToTensor()])
    return transform(image)


# cropping out some of the white pixels. This unifies the representation of images containing the same characters,
# but written in different sizes or located in different areas of the image.
def crop(image):
    if image.width < IMAGE_SIZE or image.height < IMAGE_SIZE:
        return image  # doing that for small images would give messy results after scaling up
    as_tensor = image_to_tensor(image).float()
    non_white = torch.mean(as_tensor, dim=0) != 255.0
    nonzero_indices = non_white.nonzero()
    if torch.numel(nonzero_indices) == 0:  # happens in the dataset, apparently
        return image
    min_x = torch.min(nonzero_indices[:, 1]).item()
    max_x = torch.max(nonzero_indices[:, 1]).item()
    min_y = torch.min(nonzero_indices[:, 0]).item()
    max_y = torch.max(nonzero_indices[:, 0]).item()

    # we may not want to crop the image to a smaller size than the final desired size
    max_x, min_x = pad_to_desired_size(max_x, min_x, IMAGE_SIZE)
    max_y, min_y = pad_to_desired_size(max_y, min_y, IMAGE_SIZE)
    return image.crop((min_x, min_y, max_x, max_y))  # min_x, min_y, max_x, and max_t


def preprocess(image):
    image = crop(image)
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = image_to_tensor(resized)
    tensor = tensor.float()
    tensor = torch.mean(tensor, dim=0)  # greyscale
    tensor[tensor < 200.0] = 1  # all grey pixels to ones - binarization
    tensor[tensor >= 200.0] = 0  # all white pixels to zeroes
    as_np_array =\
        cv2.erode(tensor.detach().numpy(), np.ones((2, 2), np.uint8), iterations=1) # for uniform stroke width
    tensor = torch.from_numpy(as_np_array)
    return tensor
