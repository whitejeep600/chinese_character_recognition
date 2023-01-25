import sys

import torch
from PIL import Image

from constants import IMAGE_SIZE, KERNEL_SIZE, CHANNELS, POOLING_KERNEL_SIZE, STRIDE, SAVE_DIR
from model import CharacterRecognizer
from preprocessing import preprocess
from readers import read_label_mapping, read_all_characters
from utils import get_target_device


def load_model():
    labels = read_label_mapping()
    model = CharacterRecognizer(IMAGE_SIZE, len(labels), KERNEL_SIZE, CHANNELS, POOLING_KERNEL_SIZE,
                                STRIDE)
    model.load_state_dict(torch.load(SAVE_DIR + '/model.pt', map_location=torch.device(get_target_device())))
    model.eval()
    model.to(get_target_device())
    return model


if __name__ == '__main__':
    assert len(sys.argv) == 2, "Please provide a path to an image as an argument."
    path = sys.argv[1]
    model = load_model()
    with Image.open(path) as image:
        image_tensor = preprocess(image)
    predictions = model(torch.unsqueeze(image_tensor.to(get_target_device()), 0))
    # unsqueezing to make it look like batch input (things are a bit convoluted
    # because of the convolution mechanism), of course that shouldn't be
    # necessary but I ran out of time to correct this
    prediction = torch.argmax(predictions).item()
    print(f'Predicted character: {read_all_characters()[prediction]}')


