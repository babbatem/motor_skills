import math
import numpy as np

from PIL import Image

import open3d as o3d

import numpy_point_cloud_processing as npc

def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# class that renders and processes depth images from multiple cameras,
# and combines them into point clouds
class PointCloudGenerator(object):
    def __init__(self, sim):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

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
            #math.tan(fovx/2)*self.img_height = math.tan(fovy/2)*self.img_width
            print("fovy", self.sim.model.cam_fovy[cam_id], fovy, fovx)
            fx = 1/math.tan(fovx/2.0)
            fy = 1/math.tan(fovy/2.0)
            print("fx, fy", fx, fy)
            old_fx = 0.5 * self.img_width / math.tan(fovx)
            old_fy = 0.5 * self.img_height / math.tan(fovy)
            cam_mat = np.array(((fx, 0, self.img_width / 2), (0, fy, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def generateCroppedPointCloud(self):
        single_cloud_size = self.img_width*self.img_height
        combined_cloud_and_normals = np.empty((single_cloud_size*len(self.cam_names), 3+3))
        od_clouds = []
        xy_scaler = np.array([1/1000., 1/1000., 1.])
        for cam_i in range(len(self.cam_names)):
            depth_img = self.captureImage(cam_i)
            self.saveImg(depth_img, "/home/mcorsaro/Desktop/", "depth_test_" + str(cam_i))
            color_img = self.captureImage(cam_i, False)
            self.saveImg(color_img, "/home/mcorsaro/Desktop/", "color_test_" + str(cam_i))

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

            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            od_depth = o3d.geometry.Image(depth_img)
            
            np_depth = np.asarray(od_depth)
            print("NP Depth", np_depth.min(), np_depth.max(), "\n", od_cammat.intrinsic_matrix, "\n", self.cam_mats[cam_i], "\n")

            original_od_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)#, depth_scale=0.001)

            original_np_cloud = np.asarray(original_od_cloud.points)
            scaled_np_cloud = original_np_cloud*xy_scaler
            od_cloud = o3d.geometry.PointCloud()
            od_cloud.points = o3d.utility.Vector3dVector(scaled_np_cloud)

            np_cloud = np.asarray(od_cloud.points)
            odc_min = np_cloud.min(0)[:3]
            odc_max = np_cloud.max(0)[:3]
            odc_range = [[odc_min[i], odc_max[i]] for i in range(3)]
            print("o3d cloud", np_cloud.shape, odc_range, np_cloud[0, :])

            point_cloud = self.depthToPointCloud(depth_img, cam_i)
            print("hand-made cloud", point_cloud.shape, point_cloud.min(), point_cloud.max(), point_cloud[0, :])

            od_clouds.append(od_cloud)

            if point_cloud.shape[0] != single_cloud_size:
                print("UNEXPECTED POINT CLOUD SHAPE", point_cloud.shape, "EXPECTED", single_cloud_size)
            
            tf_cloud = npc.transformCloud(point_cloud, camera_to_world_trans)
            normals = npc.estimateNormals(tf_cloud, camera_pos)
            combined_cloud_and_normals[single_cloud_size*cam_i:single_cloud_size*(cam_i+1), :3] = tf_cloud
            combined_cloud_and_normals[single_cloud_size*cam_i:single_cloud_size*(cam_i+1), 3:] = normals

        #sys.exit()
        # DON'T VISUALIZE UNTIL ALL CLOUDS ARE RENDERED - MUJOCO gets weird
        for od_cloud in od_clouds:
            o3d.visualization.draw_geometries([od_cloud])
                                  #zoom=0.3412,
                                  #front=[-0.5, -0.5, -0.5],
                                  #lookat=[0, 0, 2],
                                  #up=[-0.0694, -0.9768, 0.2024])

        # Crop using hard-coded values so only door handle (graspable area) remains
        cropped_cloud_and_normals = npc.cropPointsOutside(combined_cloud_and_normals, [None, None], [None, 0.1], [None, None])
        print("Cropped from", combined_cloud_and_normals.shape, "to", cropped_cloud_and_normals.shape)
        return cropped_cloud_and_normals

    # from babbatem's depth image rendering code
    def bufferToReal(self, z):
        self.znear = 0.1
        self.zfar = 50.#12.0
        real_depth = 2*self.zfar*self.znear / (self.zfar + self.znear - (self.zfar - self.znear)*(2*z -1))
        norm_depth = real_depth / self.zfar
        return norm_depth

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        print("enf", self.sim.model.stat.extent, self.sim.model.vis.map.znear, self.sim.model.vis.map.zfar)
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, cam_ind, capture_depth=True):
        rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=self.cam_names[cam_ind], depth=capture_depth)
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)
            real_depth = self.depthimg2Meters(depth)#norm_depth = self.bufferToReal(depth)

            return real_depth
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
