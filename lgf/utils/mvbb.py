import numpy as np
from sklearn.decomposition import PCA
from pyquaternion import Quaternion

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import trimesh

import os
import re

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise ValueError('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])

    remaining_lines = file.readlines()
    # remove empty line
    remaining_lines = [line for line in remaining_lines if line.strip() != '']

    verts = [[float(s) for s in re.split(r'\s{1,}', line.strip())] for line in remaining_lines[:n_verts]]
    faces = [[int(s) for s in re.split(r'\s{1,}', line.strip())][1:] for line in remaining_lines[n_verts:]]
    
    return verts, faces

def compute_minimum_volume_bounding_box_from_rotation(point,rotation):
    rot_vertices = np.matmul(rotation, point.transpose()).transpose()
    center = rot_vertices.max(axis=0) + rot_vertices.min(axis=0)
    center /= 2
    
    center = rotation.transpose() @ center 
    size = rot_vertices.max(axis=0) - rot_vertices.min(axis=0)

    # print('size', size)
    # print ('rot_vertices', rot_vertices)

    return center, size, rotation

def compute_minimum_volume_bounding_box_cgal(point,faces):
    mesh = trimesh.Trimesh(vertices=point, faces=faces)
    
    mesh.export('mesh/convex_hull.obj')
    os.system('./bin/optimal_bounding_box mesh/convex_hull.obj')
    
    matrix = np.loadtxt('affine_convex_hull.txt')
    
    rot = matrix[:3,:3]

    rot_vertices = np.matmul(rot, point.transpose()).transpose()
    center = rot_vertices.max(axis=0) + rot_vertices.min(axis=0)
    center /= 2
    center = rot.transpose() @ center 
    size = rot_vertices.max(axis=0) - rot_vertices.min(axis=0)

    return center, size, rot

def compute_minimum_volume_bounding_box(points):
    # print(points)
    convex_hull = ConvexHull(points)
    points = points[convex_hull.vertices]
    center = points.mean(axis=0, keepdims=True)
    points = points - center
    center = center[0, :]

    pca = PCA()
    pca.fit(points)
    pcomps = pca.components_

    points_local = np.matmul(pcomps, points.transpose()).transpose()

    size = points_local.max(axis=0) - points_local.min(axis=0)

    xdir = pcomps[0, :]
    xdir /= np.linalg.norm(xdir)
    ydir = pcomps[1, :]
    ydir /= np.linalg.norm(ydir)
    zdir = np.cross(xdir, ydir)

            
    rotmat = np.vstack([xdir, ydir, zdir]).T

    points_rot = np.matmul(rotmat, points.transpose()).transpose()
    center_shifted = center.copy()
    center = points_rot.max(axis=0) + points_rot.min(axis=0)
    center /= 2
    center = rotmat.transpose() @ center + center_shifted

    return np.hstack([center, size]).astype(np.float32), rotmat

def plot_box_and_points(center, size, rotmat,  points,  ax=None, bb_color='r', point_color='b'):

    rotmat = rotmat.copy()
    rotmat = rotmat.transpose()
    xdir = rotmat[:, 0]
    ydir = rotmat[:, 1]
    zdir = rotmat[:, 2]

    xdir *= size[0] /2
    ydir *= size[1] /2
    zdir *= size[2] /2

    corners = np.array([
        center + xdir + ydir + zdir,
        center + xdir + ydir - zdir,
        center + xdir - ydir + zdir,
        center + xdir - ydir - zdir,
        center - xdir + ydir + zdir,
        center - xdir + ydir - zdir,
        center - xdir - ydir + zdir,
        center - xdir - ydir - zdir,
    ])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_color, marker='o')
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c=bb_color, marker='o')
    ax.plot(corners[[0, 1], 0], corners[[0, 1], 1], corners[[0, 1], 2], c=bb_color)
    ax.plot(corners[[0, 2], 0], corners[[0, 2], 1], corners[[0, 2], 2], c=bb_color)
    ax.plot(corners[[0, 4], 0], corners[[0, 4], 1], corners[[0, 4], 2], c=bb_color)
    ax.plot(corners[[1, 3], 0], corners[[1, 3], 1], corners[[1, 3], 2], c=bb_color)
    ax.plot(corners[[1, 5], 0], corners[[1, 5], 1], corners[[1, 5], 2], c=bb_color)
    ax.plot(corners[[2, 3], 0], corners[[2, 3], 1], corners[[2, 3], 2], c=bb_color)
    ax.plot(corners[[2, 6], 0], corners[[2, 6], 1], corners[[2, 6], 2], c=bb_color)
    ax.plot(corners[[3, 7], 0], corners[[3, 7], 1], corners[[3, 7], 2], c=bb_color)
    ax.plot(corners[[4, 5], 0], corners[[4, 5], 1], corners[[4, 5], 2], c=bb_color)
    ax.plot(corners[[4, 6], 0], corners[[4, 6], 1], corners[[4, 6], 2], c=bb_color)
    ax.plot(corners[[5, 7], 0], corners[[5, 7], 1], corners[[5, 7], 2], c=bb_color)
    ax.plot(corners[[6, 7], 0], corners[[6, 7], 1], corners[[6, 7], 2], c=bb_color)

    ax.axis('equal')

    if ax is None:
        plt.show()
