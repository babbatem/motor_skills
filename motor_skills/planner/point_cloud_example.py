import math
import numpy as np
import open3d as o3d

def generatePointCloud():

    img_width = 640
    img_height = 480

    aspect_ratio = img_width/img_height
    # sim.model.cam_fovy[0] = 60
    fovy = math.radians(60)
    fovx = 2 * math.atan(math.tan(fovy / 2) * aspect_ratio)
    fx = 1/math.tan(fovx/2.0)
    fy = 1/math.tan(fovy/2.0)
    cx = img_width/2
    cy = img_height/2
    cam_mat = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, fx, fy, cx, cy)

    depth_img = captureImage()

    o3d_depth = o3d.geometry.Image(depth_img)
    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, cam_mat)

    o3d.visualization.draw_geometries([o3d_cloud])

# Render and process an image
def captureImage():
    #img, depth = sim.render(img_width, img_height, camera_name=sim.model.camera_names[0], depth=True)
    # 480x640 np array
    depth = np.loadtxt("/home/mcorsaro/depth_image_rendered.npy").astype(np.float32)

    flipped_depth = np.flip(depth, axis=0)
    real_depth = depthimg2Meters(flipped_depth)
    return real_depth

# https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
def depthimg2Meters(depth):
    # sim.model.stat.extent = 1.6842802984193577
    # sim.model.vis.map.znear = 0.1
    # sim.model.vis.map.zfar = 12.0
    extent = 1.6842802984193577
    near = 0.1 * extent
    far = 12. * extent
    image = near / (1 - depth * (1 - near / far))
    return image

if __name__ == '__main__':
    generatePointCloud()

def scaleCloudXY(cloud):
    xy_scaler = np.array([1/1000., 1/1000., 1.])

    np_cloud = np.asarray(cloud.points)
    scaled_np_cloud = np_cloud*xy_scaler
    scaled_cloud = o3d.geometry.PointCloud()
    scaled_cloud.points = o3d.utility.Vector3dVector(scaled_np_cloud)
    return scaled_cloud