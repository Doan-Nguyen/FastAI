from imports import * 
from configs import * 

def data_preprocess(images_path):
    frames = get_image_files(images_path)
    pat = r'/([^/]+)_\d+.jpg$'  # linux
    # pat = r'\\.+\\([^/]+)_\d+.jpg$' # win 
    data = ImageDataBunch.from_name_re(
                images_path, 
                frames, 
                pat, 
                ds_tfms= get_transforms(),
                size=224,
                bs=bs).normalize(imagenet_stats)

    return data