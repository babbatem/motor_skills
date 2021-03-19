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

        self.target_bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-500, -500, -500.), max_bound=(500., 500., 500.))

        self.cam_mats = []
        for cam_id in range(len(self.cam_names)):
            aspect_ratio = self.img_width/self.img_height
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def generateCroppedPointCloud(self):
        o3d_clouds = []
        for cam_i in range(len(self.cam_names)):
            depth_img = self.captureImage(cam_i)
            '''
            self.saveImg(depth_img, "/home/mcorsaro/Desktop/", "depth_test_" + str(cam_i))
            color_img = self.captureImage(cam_i, False)
            self.saveImg(color_img, "/home/mcorsaro/Desktop/", "color_test_" + str(cam_i))
            '''

            cam_body_id = self.sim.model.cam_bodyid[cam_i]
            camera_pos, camera_quat = self.sim.model.body_pos[cam_body_id], self.sim.model.body_quat[cam_body_id]
            '''for i in range(len(self.sim.model.body_quat)):
                print(self.sim.model.body_names[i], self.sim.model.body_quat[i])'''
            world_to_camera_trans = npc.posQuat2Mat(camera_pos, camera_quat)
            camera_to_world_trans = np.linalg.inv(world_to_camera_trans)

            print("Camera pose", camera_pos, camera_quat, "\n", camera_to_world_trans)

            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            #od_cammat = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            od_depth = o3d.geometry.Image(depth_img)
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)

            '''np_depth = np.asarray(od_depth)
            print("NP Depth", np_depth.min(), np_depth.max(), "\n", od_cammat.intrinsic_matrix, "\n", self.cam_mats[cam_i], "\n")

            original_o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)#, depth_scale=0.001)

            original_np_cloud = np.asarray(original_o3d_cloud.points)
            scaled_np_cloud = original_np_cloud*xy_scaler
            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(scaled_np_cloud)

            np_cloud = np.asarray(o3d_cloud.points)
            odc_min = np_cloud.min(0)[:3]
            odc_max = np_cloud.max(0)[:3]
            odc_range = [[odc_min[i], odc_max[i]] for i in range(3)]
            print("o3d cloud", np_cloud.shape, odc_range, np_cloud[0, :])

            point_cloud = self.depthToPointCloud(depth_img, cam_i)
            h_min = point_cloud.min(0)[:3]
            h_max = point_cloud.max(0)[:3]
            h_range = [[h_min[i], h_max[i]] for i in range(3)]
            print("hand-made cloud", point_cloud.shape, h_range, point_cloud[0, :], "\n")'''

            transformed_cloud = o3d_cloud.transform(camera_to_world_trans)
            transformed_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=250))
            transformed_cloud.orient_normals_towards_camera_location(camera_pos)

            cropped_cloud = transformed_cloud.crop(self.target_bounds)

            o3d_clouds.append(cropped_cloud)

        # DON'T VISUALIZE UNTIL ALL CLOUDS ARE RENDERED - MUJOCO gets weird
        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        
        '''n_axis_pts = 300
        axes_points = []
        axes_colors = []
        for i in range(n_axis_pts):
            axes_points.append([float(i)/n_axis_pts, 0, 0])
            axes_colors.append([255, 0, 0])
            axes_points.append([0, float(i)/n_axis_pts, 0])
            axes_colors.append([0, 255, 0])
            axes_points.append([0, 0, float(i)/n_axis_pts])
            axes_colors.append([0, 0, 255])'''

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
        for cloud in o3d_clouds:
            o3d.visualization.draw_geometries([cloud, axes])
        return combined_cloud

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
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
            real_depth = self.depthimg2Meters(depth)

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
