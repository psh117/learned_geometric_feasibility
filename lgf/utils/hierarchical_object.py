import trimesh
import numpy as np
import matplotlib.pyplot as plt

import os

from lgf.utils.convex_decomposition import convex_decomposition
from lgf.utils.mvbb import plot_box_and_points, compute_minimum_volume_bounding_box_cgal, compute_minimum_volume_bounding_box_from_rotation

class HierarchicalObject:
    def __init__(self):
        self.parts = []
        self.boxes = [] # (center, size, rotation)
        self.rotation_loaded = False
        self.mesh_vertices = None
        self.mesh_faces = None

        self.mesh_bound = None

    def load_mesh(self, mesh_file, display=False, **kwargs):
        self.mesh = trimesh.load(mesh_file)
        
        self.mesh_vertices = self.mesh.vertices
        self.mesh_faces = self.mesh.faces

        self.mesh_bound = (self.mesh.vertices.min(axis=0).view(np.ndarray), self.mesh.vertices.max(axis=0).view(np.ndarray))
        path_prefix = ''
        if mesh_file[0] == '/' or mesh_file[0] == '~':
            path_prefix = mesh_file[0]

        self.mesh_dir = mesh_file.split('/')[:-1]
        self.mesh_name = mesh_file.split('/')[-1].split('.')[0]
        # print('mesh_dir', self.mesh_dir)

        save_dir = os.path.join(*self.mesh_dir, self.mesh_name)
        save_dir = path_prefix + save_dir
        self.save_dir = save_dir
        # print('save_dir', save_dir)
        if os.path.exists(os.path.join(save_dir, 'part_0_vertices.txt')):
            # print('load from file')
            self.parts = []
            i = 0
            while True:
                part_file = os.path.join(save_dir, 'part_{}_vertices.txt'.format(i))
                if os.path.exists(part_file):
                    vertices = np.loadtxt(part_file)
                    faces = np.loadtxt(os.path.join(save_dir, 'part_{}_faces.txt'.format(i)))
                    self.parts.append((vertices, faces))
                    i += 1
                else:
                    break
                
            self.rotation = []

            self.rotation_loaded = True
            for i in range(len(self.parts)):
                if os.path.exists(os.path.join(save_dir, 'part_{}_rotation.txt'.format(i))):
                    self.rotation.append(np.loadtxt(os.path.join(save_dir, 'part_{}_rotation.txt'.format(i))))
                else:
                    self.rotation_loaded = False
                
        else:
            self.parts = convex_decomposition(self.mesh, **kwargs) 
            # save vertices and faces to file
            for i, part in enumerate(self.parts):
                os.makedirs(save_dir, exist_ok=True)
                vertices, faces = part

                np.savetxt(os.path.join(save_dir, 'part_{}_vertices.txt'.format(i)), vertices)
                np.savetxt(os.path.join(save_dir, 'part_{}_faces.txt'.format(i)), faces)

        if display:
            mesh_parts = []
            for vs, fs in self.parts:
                mesh_parts.append(trimesh.Trimesh(vs, fs))
        
            scene = trimesh.Scene()
            for p in mesh_parts:
                p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
                scene.add_geometry(p)
            scene.show()
    
    def compute_bounding_box(self, plot=False):
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        if self.rotation_loaded:
            for part, rotation in zip(self.parts, self.rotation):
                vertices, faces = part
                
                self.boxes.append(compute_minimum_volume_bounding_box_from_rotation(vertices, rotation))
                if plot:
                    plot_box_and_points(self.boxes[-1][0],self.boxes[-1][1],self.boxes[-1][2], vertices, ax=ax, bb_color='r', point_color='b')
        else:
            i = 0
            for part in self.parts:
                vertices, faces = part
                
                self.boxes.append(compute_minimum_volume_bounding_box_cgal(vertices, faces))
                np.savetxt(os.path.join(self.save_dir, 'part_{}_rotation.txt'.format(i)), self.boxes[-1][2])
                if plot:
                    plot_box_and_points(self.boxes[-1][0],self.boxes[-1][1],self.boxes[-1][2], vertices, ax=ax, bb_color='r', point_color='b')
                i += 1

        if plot:
            fig.show()

    def get_box_transform(self, part_index):
        center, size, rotation = self.boxes[part_index]

        t_box_obj = np.eye(4)
        t_box_obj[:3, 3] = center
        t_box_obj[:3, :3] = rotation.transpose() # should be transpose

        return t_box_obj
    
    def get_box_size(self, part_index):
        center, size, rotation = self.boxes[part_index]
        return size

    def get_grasp_pose(self, t_world_obj, part_index, grasp_mode, grasp_parameter, grasp_palm_depth=0.02):
        """
        grasp_mode (str): combination of {top, bottom, right, left, front, back}, {a, b}, and {1, 2}
        {top, bottom, right, left, front, back} : z axis of the gripper
        {a, b} : x axis of the gripper
        {1, 2} : symmetric gripper

        e.g. top_a1, bottom_b2, right_a2, left_b1, front_a1, back_b2

        grasp_parameter (float): 0 ~ 1
        """
        center, size, rotation = self.boxes[part_index]

        t_obj_box = self.get_box_transform(part_index)

        t_world_box = np.dot(t_world_obj, t_obj_box)
        t_box_grasp = np.eye(4)
        pos_box_grasp = np.zeros(3)
        r_box_grasp = np.zeros((3,3))
        
        # set gripper direction (z axis)
        if 'top' in grasp_mode:
            r_box_grasp[:,2] = np.array([0, 0, -1])
            pos_box_grasp[2] = max(size[2]/2 - grasp_palm_depth, 0)

        elif 'bottom' in grasp_mode:
            r_box_grasp[:,2] = np.array([0, 0, 1])
            pos_box_grasp[2] = - max(size[2]/2 - grasp_palm_depth, 0)

        elif 'right' in grasp_mode:
            r_box_grasp[:,2] = np.array([0, 1, 0])
            pos_box_grasp[1] = -max(size[1]/2 - grasp_palm_depth, 0)

        elif 'left' in grasp_mode:
            r_box_grasp[:,2] = np.array([0, -1, 0])
            pos_box_grasp[1] = max(size[1]/2 - grasp_palm_depth, 0)

        elif 'front' in grasp_mode:
            r_box_grasp[:,2] = np.array([1, 0, 0])
            pos_box_grasp[0] = - max(size[0]/2 - grasp_palm_depth, 0)

        elif 'back' in grasp_mode:
            r_box_grasp[:,2] = np.array([-1, 0, 0])
            pos_box_grasp[0] = max((size[0]/2 - grasp_palm_depth), 0)

        else:
            raise ValueError('grasp_mode should be top, bottom, right, left, front, or back')

        # set gripper x and y axis (x axis is the direction of the gripper; parallel jaws move along y axis)
        z_index = np.argmax(np.abs(r_box_grasp[:,2]))

        if 'a' in grasp_mode.split('_')[1]:
            x_index = (z_index + 1) % 3
        elif 'b' in grasp_mode.split('_')[1]:
            x_index = (z_index + 2) % 3
        else:
            raise ValueError('grasp_mode should be a or b')
        
        if '1' in grasp_mode:
            r_box_grasp[x_index, 0] = 1
        elif '2' in grasp_mode:
            r_box_grasp[x_index, 0] = -1
        else:
            raise ValueError('grasp_mode should be 1 or 2')

        pos_box_grasp[x_index] = size[x_index] * grasp_parameter - size[x_index]/2
        r_box_grasp[:,1] = np.cross(r_box_grasp[:,2], r_box_grasp[:,0])

        t_box_grasp[:3, :3] = r_box_grasp
        t_box_grasp[:3, 3] = pos_box_grasp

        t_world_grasp = np.dot(t_world_box, t_box_grasp)

        return t_world_grasp
        
    
# how to transfer grasp mode?
# find nearest grasp mode in the dataset
# no use box mode instead of grasp mode