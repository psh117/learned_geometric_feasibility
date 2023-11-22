from lgf.utils.hierarchical_object import HierarchicalObject
from lgf.utils.scene_generation import scene_generation
from lgf.utils.ros_utils import get_package_path
from srmt.planning_scene.planning_scene import PlanningSceneLight
from srmt.kinematics.trac_ik import TRACIK
from srmt.utils.transform_utils import quaternion_rotation_matrix, get_transform, get_pose
from scipy.spatial.transform import Rotation
import numpy as np
from srmt.planning_scene.visual_sim import VisualSimulator
from srmt.utils import ros_init

import matplotlib.pyplot as plt

import time
import argparse
import os
import rospy
import yaml

import xml.etree.ElementTree as ET
import pickle
import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--mesh', '-M', type=str, default='side_chair')
parser.add_argument('--num-samples', '-N', type=int, default=20000)
parser.add_argument('--save-every', '-E', type=int, default=1000)
parser.add_argument('--num-keep', '-K', type=int, default=2)
parser.add_argument('--seed', '-S', type=int, default=1107)
parser.add_argument('--grasp_attempts', '-T', type=int, default=20)
parser.add_argument('--num-grid', '-G', type=int, default=10)
parser.add_argument('--scene', '-C', type=str, default='table_middle')
# parser.add_argument('--robot', '-R', type=str, default='panda')

args = parser.parse_args()

if __name__ == '__main__':
    np.random.seed(args.seed)

    scene_config_dir = os.path.join('config', 'scenes', args.scene)
    # robot_config = yaml.load(open(os.path.join('config', 'robots', args.robot + '.yaml'), 'r'), Loader=yaml.FullLoader)

    scene_config = os.path.join(scene_config_dir, 'scene.yaml')
    scene_config = yaml.load(open(scene_config, 'r'), Loader=yaml.FullLoader)
    sensor_config = os.path.join(scene_config_dir, 'sensors.yaml')
    sensor_config = yaml.load(open(sensor_config, 'r'), Loader=yaml.FullLoader)
    variation_config = os.path.join(scene_config_dir, 'variation.yaml')
    variation_config = yaml.load(open(variation_config, 'r'), Loader=yaml.FullLoader)
    
    # load robot config
    ros_init('box_dataset_gen')
    
    # urdf_path = get_package_path(robot_config['urdf'])
    # srdf_path = get_package_path(robot_config['srdf'])

    # print(urdf_path)
    # print(srdf_path)

    # urdf = open(urdf_path).read()
    # srdf = open(srdf_path).read()

    # rospy.set_param('robot_description', urdf)
    # rospy.set_param('robot_description_semantic', srdf)

    # srdf_tree = ET.fromstring(srdf)
    # planning_group = srdf_tree.find('group[@name="panda_arm"]')
    
    # base_link = planning_group.findall('chain')[0].attrib['base_link']
    # tip_link = planning_group.findall('chain')[0].attrib['tip_link']


    grasp_lists = []
    grasps = ['top', 'bottom', 'left', 'right', 'front', 'back']
    # grasps = ['back']
    types = ['a1', 'b1']
    for grasp in grasps:
        for t in types:
            grasp_lists.append(grasp + '_' + t)


    pc = PlanningSceneLight(base_link='panda_link0')
    trac_ik = TRACIK(base_link='panda_link0', tip_link='panda_hand_tcp', max_time=0.01)

    q_init = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 1.57/2])
    pc.update_joints('panda_arm', q_init)
    pc.update_joints('panda_hand', np.array([0.035, 0.035]))

    target_object = None

    # initial loading

    target_object_dict = scene_generation(pc, scene_config, variation_config, use_variation=False, first=True)
    target_object = target_object_dict['target_object']

    # for i in range(3):
    #     pc.display()
    #     time.sleep(0.5)

    # for _ in range(1000):
    #     scene_generation(pc, scene_config, variation_config, use_variation=True, first=False)
    #     pc.display()
    #     time.sleep(0.5)
    
    if target_object['primitives'][0]['type'] == 'mesh':
        mesh_path = target_object['primitives'][0]['path']
        mesh_abs_path = get_package_path(mesh_path)

        h = HierarchicalObject()
        h.load_mesh(mesh_abs_path)
        h.compute_bounding_box()
        
    elif target_object['primitives'][0]['type'] == 'box':
        h = HierarchicalObject()
        center = np.zeros(3)
        size = np.array(target_object['primitives'][0]['dimensions'])
        rotation = np.eye(3)
        h.boxes = [(center, size, rotation)]
    
    vs = VisualSimulator()
    
    dataset_len = 0
    voxels = np.empty((args.num_samples, args.num_grid, args.num_grid, args.num_grid), dtype=np.int8)
    poses = np.empty((args.num_samples, 7), dtype=np.float32)
    sizes = np.empty((args.num_samples, 3), dtype=np.float32)
    modes = np.empty((args.num_samples, 12), dtype=np.int8)

    suc_cnt = 0
    fail_cnt = 0
    tq = tqdm.tqdm(total=args.num_samples)
    while True:
        if dataset_len >= args.num_samples:
            break
        target_object_dict = scene_generation(pc, scene_config, variation_config, 
                                              use_variation=True, first=False)
        
        if target_object_dict['dimensions'] is not None:
            size = target_object_dict['dimensions']
            
            if len(h.boxes) != 1:
                raise ValueError('len(h.boxes) != 1')
            
            center, _, rotation = h.boxes[0]
            h.boxes = [(center, size, rotation)]


        obj_pos = target_object_dict['position_dict']['TargetObject']
        obj_quat = target_object_dict['orientation_dict']['TargetObject']

        if target_object['primitives'][0]['type'] == 'mesh':
            mesh_path = target_object['primitives'][0]['path']
            mesh_abs_path = get_package_path(mesh_path)

            h = HierarchicalObject()
            h.load_mesh(mesh_abs_path)
            h.compute_bounding_box()

            for i, box in enumerate(h.boxes):
                cneter, size, rotation = box
                t_obj_box = h.get_box_transform(i)
                box_size = h.get_box_size(i)
                
                t_world_obj = np.eye(4)
                t_world_obj[:3, 3] = obj_pos
                t_world_obj[:3, :3] = Rotation.from_quat(obj_quat).as_matrix()

                t_world_box = np.dot(t_world_obj, t_obj_box)

                pos, quat = get_pose(t_world_box)

                pc.add_box('box' + str(i), box_size, pos, quat)

        vs.load_scene(pc)
        pm = np.empty((0, 3))
        for cam_point in sensor_config['cam_points']:
            for look_object in sensor_config['look_objects']:
                obj_name = look_object['name']
                look_object_position = np.array(target_object_dict['position_dict'][obj_name])
                look_object_orientation = np.array(target_object_dict['orientation_dict'][obj_name])
                look_object_orientation = Rotation.from_quat(look_object_orientation).as_matrix()
                local_translation = np.array(look_object['position'])
                t = look_object_orientation @ local_translation
                look_object_position = look_object_position + t
                vs.set_cam_and_target_pose(np.array(cam_point), look_object_position)
                pm = np.vstack((pm, vs.generate_point_cloud_matrix()))

        cam_point = np.array(sensor_config['cam_points'][0])

        for i, box in enumerate(h.boxes): # center, size, rotation
            t_obj_box = h.get_box_transform(i)
            box_size = h.get_box_size(i)
            # print('box size', box_size)
            t_world_obj = np.eye(4)
            t_world_obj[:3, 3] = obj_pos
            t_world_obj[:3, :3] = Rotation.from_quat(obj_quat).as_matrix()

            t_world_box = np.dot(t_world_obj, t_obj_box)

            box_pos, box_quat = get_pose(t_world_box)

            pos_world_box = t_world_box[:3, 3]
            quat_world_box = Rotation.from_matrix(t_world_box[:3, :3]).as_quat()
            
            voxel = vs.generate_local_voxel_occupancy(pm, box_pos, box_quat, 
                                                      cam_point, 
                                                      -box_size/2, box_size/2, 
                                                      n_grids=np.array([args.num_grid,args.num_grid,args.num_grid]), 
                                                      fill_occluded_voxels=True)
            
            voxel = voxel.astype(np.int8)

            # voxel = voxel > 0.5

            grasp_suc = np.zeros(len(grasp_lists), dtype=np.bool8)
            for j, grasp_mode in enumerate(grasp_lists):

                for _ in range(args.grasp_attempts):
                    grasp_parameter = np.random.uniform(0.05, 0.95)
                    # print('grasp_mode', grasp_mode)
                    t_world_grasp = h.get_grasp_pose(t_world_obj, i, grasp_mode, 
                                                     grasp_parameter=grasp_parameter)

                    pos = t_world_grasp[:3, 3]
                    quat = Rotation.from_matrix(t_world_grasp[:3, :3]).as_quat()
                    r, q_res = trac_ik.solve(pos, quat, q_init)
                    pc.update_joints('panda_arm', q_res)
                    # pc.display()
                    if r:
                        pc.update_joints('panda_arm', q_res)
                        if pc.is_current_valid():
                            pc.display()
                            # print('solved! grasp_mode', grasp_mode)
                            # print('box index', i)
                        
                            # input('press enter to continue')
                            grasp_suc[j] = 1
                            break
                # print(grasp_mode)
                # time.sleep(1.5)
            if grasp_suc.sum() == 0:
                fail_cnt += 1
                # pc.display()
            else:
                suc_cnt += 1
            tq.set_postfix({'suc': suc_cnt, 'fail': fail_cnt})
            #     continue
            if dataset_len >= args.num_samples:
                break

            voxels[dataset_len] = voxel
            poses[dataset_len] = np.concatenate((box_pos, box_quat))
            sizes[dataset_len] = box_size
            modes[dataset_len] = grasp_suc

            dataset_len += 1
            tq.update(1)

            if dataset_len % args.save_every == 0:
                # save dataset
                dir = f'datasets/{args.scene}'
                surfix = f'_{args.scene}_{dataset_len}_{args.seed}_{args.num_grid}'

                if not os.path.exists(dir):
                    os.makedirs(dir)
                    
                print('saving dataset at ', dataset_len)
                np.save(f'{dir}/voxels{surfix}.npy', voxels[:dataset_len])
                np.save(f'{dir}/poses{surfix}.npy', poses[:dataset_len])
                np.save(f'{dir}/sizes{surfix}.npy', sizes[:dataset_len])
                np.save(f'{dir}/modes{surfix}.npy', modes[:dataset_len])

                if dataset_len > args.num_keep * args.save_every:
                    delete_target_num = dataset_len - args.num_keep * args.save_every
                    try:
                        os.remove(f'{dir}/voxels_{args.scene}_{delete_target_num}_{args.seed}.npy')
                        os.remove(f'{dir}/poses_{args.scene}_{delete_target_num}_{args.seed}.npy')
                        os.remove(f'{dir}/sizes_{args.scene}_{delete_target_num}_{args.seed}.npy')
                        os.remove(f'{dir}/modes_{args.scene}_{delete_target_num}_{args.seed}.npy')
                    except:
                        print('failed to delete files at', delete_target_num)

            # data = {'voxel':voxel, 'box_pos':box_pos, 'box_quat':box_quat, 'box_size':box_size, 'grasp_suc':grasp_suc}
            # dataset.append(data)
            

    # save dataset
    dir = f'datasets/{args.scene}'
    surfix = f'_{args.scene}_{args.num_samples}_{args.seed}_{args.num_grid}'

    if not os.path.exists(dir):
        os.makedirs(dir)

    np.save(f'{dir}/voxels{surfix}.npy', voxels)
    np.save(f'{dir}/poses{surfix}.npy', poses)
    np.save(f'{dir}/sizes{surfix}.npy', sizes)
    np.save(f'{dir}/modes{surfix}.npy', modes)

    print('saving dataset at ', dataset_len)
    print('total dataset_len', dataset_len)