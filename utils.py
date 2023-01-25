import torch


def get_target_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_number_of_correct(predictions, languages):
    return len([i for i in range(len(predictions)) if torch.argmax(predictions[i]) == languages[i]])


# if the difference between minimum and maximum is smaller than desired_size, the function
# moves them away from each other. Since this is used for finding the boundaries of cropped images,
# minimum is not allowed to go below 0, and maximum - above desired_size.
def pad_to_desired_size(maximum, minimum, desired_size):
    size = maximum - minimum + 1
    if size < desired_size:
        white_pixels_to_add = desired_size - size
        minimum -= white_pixels_to_add // 2
        maximum += white_pixels_to_add // 2 if white_pixels_to_add % 2 == 0 else white_pixels_to_add // 2 + 1
        # we might have exceeded the boundaries of the picture this way
        if minimum < 0:
            maximum -= minimum
            minimum = 0
        if maximum > desired_size - 1:
            minimum -= maximum - (desired_size - 1)
            maximum = desired_size - 1
    return maximum, minimum
