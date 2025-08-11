import torch
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from datasets.scannet200_constants import SCANNET_COLOR_MAP_200


def get_colored_image_pca_sep(feature, name):
    import matplotlib.pyplot as plt
    # Reshape the features to [num_samples, num_features]
    w, h, d = feature.shape
    reshaped_features = feature.reshape((w * h, d))

    # Apply PCA to reduce dimensionality to 3
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped_features)

    # Normalize the PCA results to 0-1 range for visualization
    pca_result -= pca_result.min(axis=0)
    pca_result /= pca_result.max(axis=0)

    # Reshape back to the original image shape
    image_data = pca_result.reshape((w, h, 3))

    # Display and save the image
    plt.imshow(image_data)
    plt.axis('off')
    plt.savefig(f'img_{name}.jpg', bbox_inches='tight', pad_inches=0)


def get_colored_point_cloud_from_soft_labels(xyz, soft_labels, name):
    # Convert soft labels to hard labels
    hard_labels = np.argmax(soft_labels, axis=1)
    unique_labels = np.unique(hard_labels)
    # Generate a colormap with 21 distinct colors
    cmap = plt.get_cmap('tab20', len(unique_labels))  # 'tab20b' has 20 distinct colors, adjust as needed for 21
    # Map hard labels to colors using the colormap
    colors = np.array([cmap(i)[:3] for i in hard_labels])  # Extract RGB components
    # Create and color the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

    # Save the point cloud
    o3d.io.write_point_cloud(name + f'.ply', pcd)


def creat_labeled_point_cloud(points, labels, name, normals=None):
    """
    Creates a point cloud where each point is colored based on its label, and saves it to a .ply file.

    Parameters:
    - points: NumPy array of shape (N, 3) representing the point cloud.
    - labels: NumPy array of shape (N,) containing integer labels for each point.
    - name: String representing the base filename for the output .ply file.
    """
    # Step 1: Initialize the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # Step 2: Map labels to colors using the predefined color map (SCANNET_COLOR_MAP_200)
    # Normalize RGB values to the [0, 1] range as required by Open3D
    colors = np.array([SCANNET_COLOR_MAP_200.get(label + 1, (0.0, 0.0, 0.0)) for label in labels]) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Step 3: Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])

    # Step 4: Save the point cloud to a .ply file
    o3d.io.write_point_cloud(f"{name}.ply", pcd)
    print(f"Point cloud saved as {name}.ply")


def get_colored_point_cloud_pca_sep(xyz, feature, name=None):
    """N x D"""
    pca = PCA(n_components=3)
    pca_gf = pca.fit_transform(feature)
    pca_gf = (pca_gf + np.abs(pca_gf.min(0))) / (pca_gf.ptp(0) + 1e-4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(pca_gf)
    o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud(name + f'.ply', pcd)


def visualize_clusters(point_cloud, labels, name=None):
    # Generate a color map where each cluster has a unique color
    colors = np.array([SCANNET_COLOR_MAP_200.get(label, (255.0, 255.0, 255.0)) for label in labels]) / 255.0

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(name + '.ply', pcd)



def vis_points(points, name):
    # Convert PyTorch tensor to NumPy array
    point_cloud_np = points.numpy()

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    o3d.io.write_point_cloud(name + '.ply', pcd)


def visualize_multiple_point_clouds(points, feats):
    """
    Visualize a list of point clouds where each point cloud contains xyz coordinates,
    RGB values, and normals.

    Args:
        point_clouds (list): A list of point clouds where each point cloud is a tuple (xyz, feats).
                             xyz: [N, 3], coords of the point cloud
                             feats: [N, 6], including RGB (first 3) and normals (last 3)

    Returns:
        None
    """
    o3d_pcds = []
    for i in range(len(points)):
        # Create Open3D point cloud object
        xyz, feat = points[i].squeeze(0).cpu().numpy(), feats[i].squeeze(0).cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        # Set points
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Set colors (normalize RGB values to [0, 1])
        colors = feat[:, :3] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Set normals
        normals = feat[:, 3:6]
        pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([pcd])
        o3d_pcds.append(pcd)

    # Visualize all point clouds together
    o3d.visualization.draw_geometries(o3d_pcds)