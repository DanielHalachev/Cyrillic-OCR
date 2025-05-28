from PIL import Image


def resize_and_pad(img, target_height, target_width):
    width, height = img.size
    new_height = target_height
    new_width = int(width * (new_height / height))
    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

    if new_width < target_width:
        # new_img = Image.new("L", (target_width, new_height))
        new_img = Image.new("RGB", (target_width, new_height), (255, 255, 255))
        new_img.paste(img, (0, 0))
        return new_img
    elif new_width > target_width:
        return img.resize((target_width, new_height), Image.Resampling.BILINEAR)
    else:
        return img


class ResizeAndPadTransform:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        return resize_and_pad(img, self.height, self.width)
