import math
import numpy as np

import numpy_point_cloud_processing as npc

def sampleGraspPoseAtIndex(cloud, normals, index):
    kdtree->radiusSearch(search_point, 0.03, point_idx_radius_search, point_radius_squared_distance);
    for (size_t pt = 0; pt < point_idx_radius_search.size(); ++pt)
    {
        local_cloud->points.push_back(cloud->points[point_idx_radius_search[pt]]);
    }
    Eigen::Matrix3d covariance_matrix;
    Eigen::Vector4d xyz_centroid;
    pcl::compute3DCentroid(*local_cloud, xyz_centroid);
    // Compute the 3x3 covariance matrix
    pcl::computeCovarianceMatrix(*local_cloud, xyz_centroid, covariance_matrix);

