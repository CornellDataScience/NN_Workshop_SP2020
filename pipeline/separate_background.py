from pipeline.pipeline import Pipeline

# import os
# os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
from PIL import Image

import numpy as np
import cv2


class SeparateBackground(Pipeline):

    def __init__(self, dst, me_kernel=(7, 7), bg_kernel=(21, 21), desaturate=True):
        self.dst = dst
        self.me_kernel = me_kernel  # mask edges gaussian blur kernel
        self.bg_kernel = bg_kernel  # background gaussian blur kernel
        self.desaturate = desaturate  # convert background to grayscale

        super().__init__()

    def map(self, data):
        self.separate_background(data)

        return data

    def separate_background(self, data):
        if "predictions" not in data:
            return

        predictions = data["predictions"]

        # print("\nPredictions is of type {} with shape {}\n\n".format(type(predictions), predictions.shape))
        instances = predictions[0][0]

        # Sum up all the instance masks
        mask = instances >= 0.5
        mask = mask.astype("uint8")*255
        # Create 3-channels mask
        mask = np.float32(np.stack([mask, mask, mask], axis=2))

        # Apply a slight blur to the mask to soften edges
        mask = cv2.GaussianBlur(mask, self.me_kernel, 0)

        # Take the foreground input image
        foreground = (data["image"].detach().numpy()[0]).transpose(1,2,0)

        # print("\n Image is type {} and of size {}\n".format(type(foreground),foreground.shape))

        # Create a Gaussian blur for the background image
        background = cv2.GaussianBlur(np.float32(foreground), self.bg_kernel, 0)

        if self.desaturate:
            # Convert background into grayscale
            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

            # convert single channel grayscale image to 3-channel grayscale image
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

        # Convert uint8 to float
        foreground = foreground.astype(float)
        background = background.astype(float)

        # print("Forground shape is {} while mask shape is {}".format(foreground.shape, mask.shape))
        # Normalize the alpha mask to keep intensity between 0 and 1
        mask = mask.astype(float)/255.0

        # Multiply the foreground with the mask
        foreground = cv2.multiply(foreground, mask)

        # Multiply the background with ( 1 - mask )
        background = cv2.multiply(background, 1.0 - mask)

        # Add the masked foreground and background
        dst_image = cv2.add(foreground, background)

        # Return a normalized output image for display
        final_dst_image = Image.fromarray(dst_image.astype("uint8")).resize(data["shape"][:2], Image.BILINEAR)
        data[self.dst] = np.array(final_dst_image)
        # print("\nGot to the end\n")
