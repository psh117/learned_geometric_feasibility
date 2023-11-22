from srmt.planning_scene.planning_scene import PlanningSceneLight
from srmt.utils.transform_utils import quaternion_rotation_matrix, get_transform, get_pose

from scipy.spatial.transform import Rotation
import numpy as np

def scene_generation(pc : PlanningSceneLight, scene_config, variation_config, use_variation=False, first=False, min_box_side_size=0.07):
    robot_offset = scene_config['robot']['base_offset']
    position_offset = np.array(robot_offset['position'])
    orientation_offset = np.array(robot_offset['orientation'])
    orientation_offset = Rotation.from_quat(orientation_offset).as_matrix()

    t_world_robot = np.eye(4)
    t_world_robot[:3, 3] = position_offset
    t_world_robot[:3, :3] = orientation_offset

    t_world_offset = np.eye(4)

    target_object_dict = {}
    target_object_dimensions = None

    position_dict = {}
    orientation_dict = {}

    if use_variation:
        for variation in variation_config:
            if 'World' in variation['names']:
                world_position_var = np.array(variation['position'])
                world_orientation_var = np.array(variation['orientation'])
                
                t_world_offset[:3, 3] = np.random.uniform(-world_position_var, world_position_var)
                euler_angles = np.random.uniform(-world_orientation_var, world_orientation_var)
                t_world_offset[:3, :3] = Rotation.from_euler('xyz', euler_angles).as_matrix()

    for obj in scene_config['world']['collision_objects']:
        position = None
        orientation = None
        dimensions_var = None

        t_obj_var = np.eye(4)

        if use_variation: 
            for variation in variation_config:
                if obj['id'] in variation['names']:
                    is_var = True
                    position_var = np.array(variation['position'])
                    orientation_var = np.array(variation['orientation'])
                    var_type = variation['type']
                    # currently only support uniform type
            
                    if var_type == 'uniform':
                        t_obj_var[:3, 3] = np.random.uniform(-position_var, position_var)
                        euler_angles = np.random.uniform(-orientation_var, orientation_var)
                        t_obj_var[:3, :3] = Rotation.from_euler('xyz', euler_angles).as_matrix()

                        if 'size' in variation:

                            size_var = np.array(variation['size'])

                            dimensions_var = np.random.uniform(-size_var, size_var)

                    break
                
        for primitive, pose in zip(obj['primitives'], obj['primitive_poses']):
            position = np.array(pose['position'])
            orientation = np.array(pose['orientation'])

            t_robot_obj = get_transform(position, orientation)
            t_world_obj = t_world_offset @ t_world_robot @ t_robot_obj @ t_obj_var

            position, orientation = get_pose(t_world_obj)

            position_dict[obj['id']] = position
            orientation_dict[obj['id']] = orientation

            if dimensions_var is not None:
                dimensions_var += primitive['dimensions']
                dimensions_var = np.clip(dimensions_var, 0.01, None)

                if obj['id'] == 'TargetObject':
                    while True:
                        if np.min(dimensions_var) < min_box_side_size:
                            break

                        dimensions_var = np.random.uniform(-size_var, size_var) + primitive['dimensions']

            # if not first:
            #     pc.remove_object(obj['id'])
            
            if not first and not primitive['type'] == 'mesh':
                pc.remove_object(obj['id'])

            if primitive['type'] == 'mesh':
                mesh_path = primitive['path']
                # pc.add_mesh(obj['id'], mesh_path, position, orientation)

            elif primitive['type'] == 'box':
                if dimensions_var is not None:
                    dimensions = dimensions_var
                else:
                    dimensions = np.array(primitive['dimensions'])
                pc.add_box(obj['id'], dimensions, position, orientation)

            elif primitive['type'] == 'cylinder':
                if dimensions_var is not None:
                    dimensions = dimensions_var
                else:
                    dimensions = np.array(primitive['dimensions'])
                pc.add_cylinder(obj['id'], dimensions[0], dimensions[1], position, orientation)

            elif primitive['type'] == 'sphere':
                if dimensions_var is not None:
                    dimensions = dimensions_var
                else:
                    dimensions = np.array(primitive['dimensions'])
                pc.add_sphere(obj['id'], dimensions[0], position, orientation)

        if obj['id'] == 'TargetObject':
            if dimensions_var is not None:
                target_object_dimensions = dimensions_var

            target_object_dict['target_object'] = obj

    target_object_dict['dimensions'] = target_object_dimensions
    target_object_dict['position_dict'] = position_dict
    target_object_dict['orientation_dict'] = orientation_dict

    return target_object_dict