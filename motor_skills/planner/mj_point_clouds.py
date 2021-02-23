import numpy as np
from PIL import Image

# class that renders and processes depth images from multiple cameras,
# and combines them into point clouds
class PointCloudGenerator(object):
    def __init__(self, sim, camera_names):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        self.znear = 0.1
        self.zfar = 50.#12.0

        # I think these can be set to anything
        self.img_width = 640
        self.img_height = 480

        self.cam_names = camera_names

    # from babbatem's depth image rendering code
    def buffer_to_real(self, z):
        return 2*self.zfar*self.znear / (self.zfar + self.znear - (self.zfar - self.znear)*(2*z -1))

    def vertical_flip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def capture_image(self, cam_ind, capture_depth=True):
        rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=self.cam_names[cam_ind], depth=capture_depth)
        if capture_depth:
            img, depth = rendered_images
            depth = self.vertical_flip(depth)
            real_depth = self.buffer_to_real(depth)
            norm_depth = real_depth / self.zfar

            return norm_depth
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.vertical_flip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def save_depth_img(self, img, filepath, filename):
        normalized_image = img/img.max()*255
        normalized_image = normalized_image.astype(np.uint8)
        im = Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".jpg")


    def depth_to_point_cloud(self):
        pass
