import math
import numpy as np

from PIL import image

import open3d as o3d

import numpy_point_cloud_processing as npc

# class that renders and processes depth images from multiple cameras,
# and combines them into point clouds
class PointCloudGenerator(object):
    def __init__(self, sim):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        self.znear = 0.1
        self.zfar = 50.#12.0

        # I think these can be set to anything
        self.img_width = 640
        self.img_height = 480

        self.cam_names = self.sim.model.camera_names

        self.cam_mats = []
        for cam_id in range(len(self.cam_names)):
            # https://github.com/openai/mujoco-py/issues/271
            aspect_ratio = self.img_width/self.img_height
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            fovx = 2 * math.atan(math.tan(fovy / 2) * aspect_ratio)
            fx = 0.5 * self.img_width / math.tan(fovx * math.pi / 360)
            fy = 0.5 * self.img_height / math.tan(fovy * math.pi / 360)
            cam_mat = np.array(((fx, 0, self.img_width / 2), (0, fy, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def generateCroppedPointCloud(self):
        single_cloud_size = self.img_width*self.img_height
        combined_cloud_and_normals = np.empty((single_cloud_size*len(self.cam_names), 3+3))
        for cam_i in range(len(self.cam_names)):
            depth_img = self.captureImage(cam_i)
            #self.saveImg(depth_img, "/home/mcorsaro/Desktop/", "dtest_" + str(cam_i))
            #color_img = self.captureImage(cam_i, False)
            #self.saveImg(color_img, "/home/mcorsaro/Desktop/", "ctest_" + str(cam_i))

            cam_body_id = self.sim.model.cam_bodyid[cam_i]
            camera_pos, camera_quat = self.sim.model.body_pos[cam_body_id], self.sim.model.body_quat[cam_body_id]
            world_to_camera_trans = npc.posQuat2Mat(camera_pos, camera_quat)
            camera_to_world_trans = np.linalg.inv(world_to_camera_trans)
            
            '''print("\nOriginal tf matrix\n", world_to_camera_trans)
            w2c_rot = world_to_camera_trans[:3,:3]
            w2c_pos = world_to_camera_trans[:3, 3]
            print("Original tf rot\n", w2c_rot)
            print("Original tf pos\n", w2c_pos)
            inv_pos = np.matmul(-1*np.linalg.inv(w2c_rot), w2c_pos)
            print("Inverted pos\n", inv_pos)
            print("Calculated mat\n", camera_to_world_trans)'''

            #od_depth = o3d.geometry.Image(depth_img)
            #od_cammat = o3d.camera.PinholeCameraIntrinsic(self.cam_mats[cam_i])
            #od_cloud = o3d.geometry.create_point_cloud_from_depth_image(od_depth, od_cammatyy)
            #sys.exit()

            point_cloud = self.depthToPointCloud(depth_img, cam_i)
            if point_cloud.shape[0] != single_cloud_size:
                print("UNEXPECTED POINT CLOUD SHAPE", point_cloud.shape, "EXPECTED", single_cloud_size)
            
            tf_cloud = npc.transformCloud(point_cloud, camera_to_world_trans)
            normals = npc.estimateNormals(tf_cloud, camera_pos)
            combined_cloud_and_normals[single_cloud_size*cam_i:single_cloud_size*(cam_i+1), :3] = tf_cloud
            combined_cloud_and_normals[single_cloud_size*cam_i:single_cloud_size*(cam_i+1), 3:] = normals
        # Crop using hard-coded values so only door handle (graspable area) remains
        cropped_cloud_and_normals = npc.cropPointsOutside(combined_cloud_and_normals, [None, None], [None, 0.1], [None, None])
        print("Cropped from", combined_cloud_and_normals.shape, "to", cropped_cloud_and_normals.shape)
        return cropped_cloud_and_normals

    # from babbatem's depth image rendering code
    def bufferToReal(self, z):
        return 2*self.zfar*self.znear / (self.zfar + self.znear - (self.zfar - self.znear)*(2*z -1))

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, cam_ind, capture_depth=True):
        rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=self.cam_names[cam_ind], depth=capture_depth)
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)
            real_depth = self.bufferToReal(depth)
            norm_depth = real_depth / self.zfar

            return norm_depth
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img/img.max()*255
        normalized_image = normalized_image.astype(np.uint8)
        im = Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".jpg")

    '''
    # Though this should work, use open3d
    def depthToPointCloud(self, depth_image, cam_id):
        # https://stackoverflow.com/questions/31265245/extracting-3d-coordinates-given-2d-image-points-depth-map-and-camera-calibratio
        point_cloud = np.zeros((depth_image.shape[0]*depth_image.shape[1], 3))
        for i in range(depth_image.shape[0]):
            for j in range(depth_image.shape[1]):
                x = (i - self.cam_mats[cam_id][0,2]) * depth_image[i,j] / self.cam_mats[cam_id][0,0]
                y = (j - self.cam_mats[cam_id][1,2]) * depth_image[i,j] / self.cam_mats[cam_id][1,1]
                z = depth_image[i,j]
                point_cloud[j+i*depth_image.shape[1]] = [x,y,z]
        return point_cloud
    '''
