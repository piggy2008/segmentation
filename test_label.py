import os
from PIL import Image
import numpy as np

path = 'river/train/labels'

images = os.listdir(path)
images.sort()
for img in images:
    image = Image.open(os.path.join(path, img)).convert('L')
    img_arr = np.array(image)
    print(img, ':', np.unique(img_arr))