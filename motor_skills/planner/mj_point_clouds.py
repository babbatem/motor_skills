import math
import numpy as np

from PIL import Image

import open3d as o3d

"""
Generates transformation matrix from position and quaternion

@param pos:  x-y-z position tuple
@param quat: w-x-y-z quaternion rotation tuple

@return trans_mat: 4x4 transformation matrix
"""
def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

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

        self.target_bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-500, -500, -500.), max_bound=(500., 500., 2.))

        self.cam_mats = []
        for cam_id in range(len(self.cam_names)):
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def generateCroppedPointCloud(self):
        o3d_clouds = []
        cam_poses = []
        for cam_i in range(len(self.cam_names)):
            depth_img = self.captureImage(cam_i)
            '''
            self.saveImg(depth_img, "/home/mcorsaro/Desktop/", "depth_test_" + str(cam_i))
            color_img = self.captureImage(cam_i, False)
            self.saveImg(color_img, "/home/mcorsaro/Desktop/", "color_test_" + str(cam_i))
            '''

            cam_body_id = self.sim.model.cam_bodyid[cam_i]
            camera_pos, camera_quat = self.sim.model.body_pos[cam_body_id], self.sim.model.body_quat[cam_body_id]
            world_to_camera_trans = posRotMat2Mat(camera_pos, quat2Mat(camera_quat))
            #posQuat2Mat(camera_pos, camera_quat)#np.matmul(posQuat2Mat(camera_pos, camera_quat), posQuat2Mat([0, 0, 0], [0, 0, 1, 0]))
            camera_to_world_trans = np.linalg.inv(world_to_camera_trans)

            other_trans = posRotMat2Mat(self.sim.model.cam_poscom0[cam_i], rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i]))
            #print("inv world_trans", camera_to_world_trans, "inv cam_mat0", np.linalg.inv(other_trans))
            #sys.exit()
            #print("\nCamera pose", camera_pos, camera_quat, "\n", world_to_camera_trans,"\n",camera_to_world_trans,"\n", other_trans)


            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            #od_cammat = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            od_depth = o3d.geometry.Image(depth_img)
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)

            '''
            np_cloud = np.asarray(o3d_cloud.points)
            odc_min = np_cloud.min(0)[:3]
            odc_max = np_cloud.max(0)[:3]
            odc_range = [[odc_min[i], odc_max[i]] for i in range(3)]
            print("o3d cloud", np_cloud.shape, odc_range, np_cloud[0, :])
            '''

            #no_base_cam = np.matmul(np.linalg.inv(other_trans), posRotMat2Mat([0, 0, 0], quat2MatList([0, 1, 0, 0])))
            '''
            handmade_rot = np.linalg.inv(posRotMat2Mat([-0.1, 0.1, 0.5], \
                np.matmul(rotMatList2NPRotMat(\
                    [1, 0, 0, \
                    0, -1, 0, \
                    0, 0, -1]) ,\
                    rotMatList2NPRotMat(\
                    [1, 0, 0, \
                    0, 1, 0, \
                    0, 0, 1]))))
            '''
            w2c_p = self.sim.model.body_pos[cam_body_id]
            b2c_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])
            '''b2c_r = rotMatList2NPRotMat(\
                    [1, 0, 0, \
                    0, 1, 0, \
                    0, 0, 1])'''
            w2b_r = rotMatList2NPRotMat(\
                    [1, 0, 0, \
                    0, -1, 0, \
                    0, 0, -1])
            w2c_r = np.matmul(b2c_r, w2b_r)
            w2c = posRotMat2Mat(w2c_p, w2c_r)
            c2w = np.linalg.inv(w2c)
            print("Using\n", w2c)
            transformed_cloud = o3d_cloud.transform(w2c)
            #print("\nother\n", other_trans, "\nno_base_cam\n", no_base_cam, "\n")
            transformed_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=250))
            transformed_cloud.orient_normals_towards_camera_location(camera_pos)

            cropped_cloud = transformed_cloud.crop(self.target_bounds)

            o3d_clouds.append(cropped_cloud)
            '''mat_tf = posQuat2Mat([0, 0, 0], [0, 0, 1, 0])
            tf_p = np.matmul(camera_to_world_trans, mat_tf)
            cam_poses.append(tf_p)
            print("tf_p\n", tf_p, "\nmat_tf\n", mat_tf)'''

        # DON'T VISUALIZE UNTIL ALL CLOUDS ARE RENDERED - MUJOCO gets weird
        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([combined_cloud, axes])
        '''for cloud in o3d_clouds:
            #world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
            #world_axes.transform(cam_poses[i])
            o3d.visualization.draw_geometries([cloud, axes])
        return combined_cloud'''

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
