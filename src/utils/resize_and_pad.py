from PIL import Image


def resize_and_pad(img, target_height, target_width):
    width, height = img.size
    new_height = target_height
    new_width = int(width * (new_height / height))
    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

    if new_width < target_width:
        new_img = Image.new("L", (target_width, new_height))
        new_img.paste(img, (0, 0))
        return new_img
    elif new_width > target_width:
        return img.resize((target_width, new_height), Image.Resampling.BILINEAR)
    else:
        return img
