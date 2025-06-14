"""
Given mesh, render canonical views with Blender
"""

import bpy
import random
import sys
from mathutils import Vector
import os
import glob
import math
import json
import mathutils
from mathutils import Matrix, Vector, Euler
from typing import List, Optional
import numpy as np


META_FILENAME = "meta.json"
RESOLUTION = 512
RENDER_SAMPLES = 64
ENGINE = "EEVEE"
CLOSE_SHADOW = False
RENDER_DEPTH = True

SUB_MESH_RENDER = "highlight"


if os.environ.get("ENGINE"):
    ENGINE = os.environ.get("ENGINE")

if os.environ.get("RENDER_SAMPLES"):
    RENDER_SAMPLES = int(os.environ.get("RENDER_SAMPLES"))

if os.environ.get("RESOLUTION"):
    RESOLUTION = int(os.environ.get("RESOLUTION"))

def generate_random(left, right):
    while True:
        val = random.gauss(0, (right-left)/5)
        if val >= left and val <= right:
            return val


def generate_frames(name):
    render_type_list = [
        {'name': "depth", "suffix": ".exr", "enable": RENDER_DEPTH},
        {'name': "render_opaque", "suffix": ".webp",
            "enable": RENDER_DEPTH},
    ]
    param = [
        {
            "type": "render",
            "name": "{}",
            "height": RESOLUTION,
            "width": RESOLUTION,
        }
    ]
    variables = ["render_"+name+".webp"]

    for render_type in render_type_list:
        if render_type["enable"]:
            param.append({
                "type": render_type["name"],
                "name": "{}",
                "height": RESOLUTION,
                "width": RESOLUTION,
            })
            variables.append(render_type["name"] +
                             "_"+name+render_type["suffix"])

    for i in range(len(variables)):
        param[i]["name"] = variables[i]
    return param


def build_transformation_mat(translation,
                             rotation) -> np.ndarray:
    """ Build a transformation matrix from translation and rotation parts.

    :param translation: A (3,) vector representing the translation part.
    :param rotation: A 3x3 rotation matrix or Euler angles of shape (3,).
    :return: The 4x4 transformation matrix.
    """
    translation = np.array(translation)
    rotation = np.array(rotation)

    mat = np.eye(4)
    if translation.shape[0] == 3:
        mat[:3, 3] = translation
    else:
        raise RuntimeError(
            f"Translation has invalid shape: {translation.shape}. Must be (3,) or (3,1) vector.")
    if rotation.shape == (3, 3):
        mat[:3, :3] = rotation
    elif rotation.shape[0] == 3:
        mat[:3, :3] = np.array(Euler(rotation).to_matrix())
    else:
        raise RuntimeError(f"Rotation has invalid shape: {rotation.shape}. Must be rotation matrix of shape "
                           f"(3,3) or Euler angles of shape (3,) or (3,1).")

    return mat


def reset_keyframes() -> None:
    """ Removes registered keyframes from all objects and resets frame_start and frame_end """
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0
    for a in bpy.data.actions:
        bpy.data.actions.remove(a)


def get_local2world_mat(blender_obj) -> np.ndarray:
    """ Returns the pose of the object in the form of a local2world matrix.
    :return: The 4x4 local2world matrix.
    """
    obj = blender_obj
    # Start with local2parent matrix (if obj has no parent, that equals local2world)
    matrix_world = obj.matrix_basis

    # Go up the scene graph along all parents
    while obj.parent is not None:
        # Add transformation to parent frame
        matrix_world = obj.parent.matrix_basis @ obj.matrix_parent_inverse @ matrix_world
        obj = obj.parent

    return np.array(matrix_world)


def add_camera(cam2world_matrix,camera_params) -> int:
    if not isinstance(cam2world_matrix, Matrix):
        cam2world_matrix = Matrix(cam2world_matrix)

    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam_ob = bpy.context.object
    cam_ob.matrix_world = cam2world_matrix
    cam_ob_data = cam_ob.data
    cam_ob_data.type = camera_params['camera_type']
    cam_ob_data.sensor_width = camera_params['camera_sensor_width']
    if camera_params['camera_type'] == 'ORTHO':
        cam_ob_data.ortho_scale = camera_params['camera_ortho_scale']
    elif camera_params['camera_type'] == 'PERSP':
        cam_ob_data.lens = camera_params['camera_lens']


def add_camera_pose(cam2world_matrix, camera_params) -> int:
    if not isinstance(cam2world_matrix, Matrix):
        cam2world_matrix = Matrix(cam2world_matrix)

    cam_ob = bpy.context.scene.camera
    cam_ob.matrix_world = cam2world_matrix
    cam_ob_data = cam_ob.data
    cam_ob_data.type = camera_params['camera_type']
    cam_ob_data.sensor_width = camera_params['camera_sensor_width']
    if camera_params['camera_type'] == 'ORTHO':
        cam_ob_data.ortho_scale = camera_params['camera_ortho_scale']
    elif camera_params['camera_type'] == 'PERSP':
        cam_ob_data.lens = camera_params['camera_lens']

    frame = bpy.context.scene.frame_end
    if bpy.context.scene.frame_end < frame + 1:
        bpy.context.scene.frame_end = frame + 1

    cam_ob.keyframe_insert(data_path='location', frame=frame)
    cam_ob.keyframe_insert(data_path='rotation_euler', frame=frame)
    cam_ob_data.keyframe_insert(data_path='type', frame=frame)
    cam_ob_data.keyframe_insert(data_path='sensor_width', frame=frame)

    if camera_params['camera_type'] == 'ORTHO':
        cam_ob_data.keyframe_insert(data_path='ortho_scale', frame=frame)
    elif camera_params['camera_type'] == 'PERSP':
        cam_ob_data.keyframe_insert(data_path='lens', frame=frame)
    return frame


def clear_normal_map():
    for material in bpy.data.materials:
        material.use_nodes = True
        node_tree = material.node_tree
        try:
            bsdf = node_tree.nodes["Principled BSDF"]
            if bsdf.inputs["Normal"].is_linked:
                for link in bsdf.inputs["Normal"].links:
                    node_tree.links.remove(link)
        except:
            pass


def enable_depth_output(output_dir: Optional[str] = '', file_prefix: str = "depth_"):

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    links = tree.links

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new('CompositorNodeRLayers')
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_z = True

    depth_output = tree.nodes.new('CompositorNodeOutputFile')
    depth_output.base_path = output_dir
    depth_output.name = 'DepthOutput'
    depth_output.format.file_format = 'OPEN_EXR'
    depth_output.format.color_depth = '32'
    depth_output.file_slots.values()[0].path = file_prefix

    links.new(rl.outputs["Depth"], depth_output.inputs['Image'])


def render():
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    links = tree.links

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new('CompositorNodeRLayers')
    else:
        rl = tree.nodes["Render Layers"]
    if bpy.context.scene.frame_end != bpy.context.scene.frame_start:
        bpy.context.scene.frame_end -= 1
        bpy.ops.render.render(animation=True, write_still=True)
        bpy.context.scene.frame_end += 1
    else:
        raise RuntimeError("No camera poses have been registered, therefore nothing can be rendered. A camera "
                           "pose can be registered via bproc.camera.add_camera_pose().")


def convert_position(location, center):
    position = ""
    axis = ['x', 'y', 'z']
    sub = location-center
    for i in range(len(axis)):
        if sub[i] > 0:
            position = position + "+" + axis[i]
        elif sub[i] < 0:
            position = position + "-" + axis[i]
    return position


def set_color_output(output_dir: Optional[str] = '', file_prefix: str = "render_"):
    scene = bpy.context.scene
    scene.render.use_compositing = True
    scene.use_nodes = True
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.image_settings.file_format = 'WEBP'
    scene.render.image_settings.quality = 100
    scene.render.image_settings.color_mode = 'RGBA'
    # scene.render.image_settings.color_depth = '16'
    scene.render.film_transparent = True
    scene.render.filepath = os.path.join(output_dir, file_prefix)
    pass


def eevee_init():
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = RENDER_SAMPLES
    if CLOSE_SHADOW == False:
        bpy.context.scene.eevee.use_gtao = True
        bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.render.use_high_quality_normals = True


def clear_scene(NOT_CLEAR_LIGHT=False):
    bpy.ops.object.select_all(action="DESELECT")
    if NOT_CLEAR_LIGHT:
        for obj in bpy.data.objects:
            if obj.type not in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)
    else:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree

    for node in node_tree.nodes:
        node_tree.nodes.remove(node)
    reset_keyframes()


def import_models(filepath, types):
    if types == "glb":
        bpy.ops.import_scene.gltf(
            filepath=filepath)
    elif types == "obj":
        forward_axis = os.environ.get('FORWARD_AXIS', 'NEGATIVE_Z')
        up_axis = os.environ.get('UP_AXIS', 'Y')
        bpy.ops.wm.obj_import(filepath=filepath, directory=os.path.dirname(filepath),forward_axis=forward_axis, up_axis=up_axis)


def rotation_matrix(x_left, x_right, y_left, y_right):
    x_rotation = math.radians(generate_random(x_left, x_right))
    y_rotation = math.radians(generate_random(y_left, y_right))
    x_rotation_matrix = mathutils.Matrix.Rotation(x_rotation, 4, 'X')
    y_rotation_matrix = mathutils.Matrix.Rotation(y_rotation, 4, 'Y')
    final_rotation_matrix = y_rotation_matrix @ x_rotation_matrix
    return final_rotation_matrix


def scene_bbox(objects=None, ignore_small_obj=False, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in objects:
        # print(max(obj.dimensions*100))
        if max(obj.dimensions*100) < 0.1 and ignore_small_obj:
            print("ignore_small_obj", obj.name,max(obj.dimensions*100))
            continue
        found = True
        for coord in obj.bound_box:
            # print(coord[0], coord[1], coord[2])
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            
            bbox_min = Vector(
                (min(bbox_min[i], coord[i]) for i in range(3)))
            bbox_max = Vector(
                (max(bbox_max[i], coord[i]) for i in range(3)))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def set_global_light(env_light=0.5):
    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes["Background"]
    back_node.inputs["Color"].default_value = Vector(
        [env_light, env_light, env_light, 1.0]
    )
    back_node.inputs["Strength"].default_value = 1.0


def normalize_scene(normalization_range, objects):
    bpy.ops.object.empty_add(type='PLAIN_AXES')
    root_object = bpy.context.object
    for obj in scene_root_objects():
        if obj != root_object:
            _matrix_world = obj.matrix_world.copy()
            obj.parent = root_object
            obj.matrix_world = _matrix_world
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox(objects)    
    scale = normalization_range / max(bbox_max - bbox_min)
    root_object.scale *= scale
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox(objects,True)
    mesh_offset = - (bbox_min + bbox_max) / 2
    root_object.matrix_local.translation = mesh_offset
    bpy.context.view_layer.update()

    bpy.ops.object.select_all(action="DESELECT")
    return root_object, bbox_max - bbox_min, scale, mesh_offset


def compute_bounding_box(mesh_objects):
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in mesh_objects:
        matrix_world = obj.matrix_world
        mesh = obj.data

        for vert in mesh.vertices:
            global_coord = matrix_world @ vert.co

            min_coords = Vector(
                (min(min_coords[i], global_coord[i]) for i in range(3)))
            max_coords = Vector(
                (max(max_coords[i], global_coord[i]) for i in range(3)))

    bbox_center = (min_coords + max_coords) / 2
    bbox_size = max_coords - min_coords

    return bbox_center, bbox_size


def change_material_blend_mode():
    for material in bpy.data.materials:
        material.use_nodes = True
        node_tree = material.node_tree
        material.blend_method = 'OPAQUE'

def change_material_blend_show_transparent(value):
    for material in bpy.data.materials:
        material.use_nodes = True
        if material.blend_method == 'BLEND':
            material.show_transparent_back = value


def get_random_points_on_sphere(center, radius, num_points):
    points = []
    for i in range(18):

        r = radius
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, 0.5*math.pi)

        x = center[0] + r * math.sin(phi) * math.cos(theta)
        y = center[1] + r * math.sin(phi) * math.sin(theta)
        z = center[2] + r * math.cos(phi)

        flag = -1 if bool(random.randint(0, 1)) else 1
        points.append(Vector((x, y, z*flag)))
    for i in range(12):

        r = radius * 0.5
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, 0.5*math.pi)

        x = center[0] + r * math.sin(phi) * math.cos(theta)
        y = center[1] + r * math.sin(phi) * math.sin(theta)
        z = center[2] + r * math.cos(phi)

        flag = -1 if bool(random.randint(0, 1)) else 1
        points.append(Vector((x, y, z*flag)))

    return points


def get_solid_points_on_sphere(center, radius):
    points = []
    elev_list = [25, 40, 25, 40, 25, 40, 25, 40, -25, -40, -25, -40, -25, -40, -25, -40]
    azim_list = [0., 45., 90., 135., 180., 225., 270., 315., 0., 45., 90., 135., 180., 225., 270., 315.]

    for i in range(16):
        x = center[0] + radius * math.cos(math.radians(elev_list[i])) * math.cos(math.radians(azim_list[i]))
        y = center[1] + radius * math.cos(math.radians(elev_list[i])) * math.sin(math.radians(azim_list[i]))
        z = center[2] + radius * math.sin(math.radians(elev_list[i]))
        points.append(Vector((x, y, z)))

    return points


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def process(filepath, types, output_path, faces_to_render=None):
    
    random.seed()
    eevee_init()
    clear_scene()
    import_models(filepath, types)
    reset_keyframes()

    bpy.ops.object.select_by_type(type='MESH')
    os.makedirs(output_path, exist_ok=True)

    mesh_objects = []  
    mesh_face_counts = []
    total_face_count = 0

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.visible_get() == True and obj.hide_get() == False:
            mesh_objects.append(obj)
            bpy.ops.object.select_all(action="DESELECT")
            bpy.ops.object.select_pattern(pattern=obj.name)

            mesh = obj.data
            num_faces = len(mesh.polygons)
            mesh_face_counts.append(num_faces)
            total_face_count += num_faces
            # print(num_faces)
            # print(obj.name)


    face_index_offset = 0  # Initialize face index offset
    # print(faces_to_render)
    # print(f"{total_face_count = }")
    for num_faces, obj in zip(mesh_face_counts, mesh_objects):
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = np.deg2rad(30)

        # Select specific faces if faces_to_render is provided
        if faces_to_render is not None:
            bpy.ops.object.mode_set(mode='OBJECT') # make sure we're in object mode
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT') # enter edit mode
            bpy.ops.mesh.select_all(action='DESELECT') # deselect everything
            bpy.ops.object.mode_set(mode='OBJECT')

            mesh = obj.data


            selected_polygons = []
            for face_index in faces_to_render:
                if face_index < face_index_offset or face_index >= face_index_offset + num_faces:
                    # print(f"Warning: Face index {face_index} is out of range.")
                    # exit()
                    continue
                mesh.polygons[int(face_index - face_index_offset)].select = True # select face
                selected_polygons.append(int(face_index - face_index_offset))

            if SUB_MESH_RENDER == "highlight": # Set faces to red
                dematerial_name = "DeselectedFacesMaterial"
                dematerial = bpy.data.materials.get(dematerial_name)
                if dematerial is None:
                    dematerial = bpy.data.materials.new(name=dematerial_name)
                    dematerial.use_nodes = True  # Enable nodes for more advanced materials
                    while dematerial.node_tree.nodes:
                        dematerial.node_tree.nodes.remove(dematerial.node_tree.nodes[0])

                    # set up white material.
                    white_shader = dematerial.node_tree.nodes.new(type='ShaderNodeEmission')
                    white_shader.inputs['Color'].default_value = (1, 1, 1, 1)
                    output_node = dematerial.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
                    dematerial.node_tree.links.new(white_shader.outputs['Emission'], output_node.inputs['Surface'])

                material_name = "SelectedFacesMaterial"
                material = bpy.data.materials.get(material_name)
                if material is None:
                    material = bpy.data.materials.new(name=material_name)
                    material.use_nodes = True  # Enable nodes for more advanced materials
                    # Customize material properties here (e.g., color)
                    principled_bsdf = material.node_tree.nodes["Principled BSDF"]
                    principled_bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)  # Red color

                obj.data.materials.append(dematerial)
                # obj.active_material = dematerial
                obj.data.materials.append(material)
                for idx in selected_polygons:
                    obj.data.polygons[idx].material_index = len(obj.data.materials) -1
            elif SUB_MESH_RENDER == "delete": # Delete selected faces
                bpy.ops.object.mode_set(mode = 'EDIT') 
                bpy.ops.mesh.delete(type='FACE')
                bpy.ops.object.mode_set(mode = 'OBJECT')
            

        face_index_offset += num_faces


    for obj in bpy.data.objects:
        if obj.animation_data is not None:
            obj.animation_data_clear()

    clear_normal_map()
    change_material_blend_show_transparent(False)


    normalization_range = 1.0
    root_object, bbox_size, scale, mesh_offset = normalize_scene(normalization_range,mesh_objects)
    bpy.context.view_layer.update()
    root_object.rotation_euler[2] = math.radians(int(os.environ.get("FORCE_ROTATION", 0)))
    bbox_center = Vector((0,0,0))

    bpy.ops.object.camera_add(location=(0, 0, 0))
    bpy.context.scene.camera = bpy.context.object


    default_camera_lens = 50
    default_camera_senser_width = 36
    default_camera_ortho_scale = 1.4

    ratio = 1
    distance = ratio * default_camera_lens / default_camera_senser_width * \
        math.sqrt(bbox_size.x**2 + bbox_size.y**2+bbox_size.z**2)
    idx = 0

    env_texture = "null"
    set_global_light(env_light=0.5)

    camera_angle_x = 2.0*math.atan(default_camera_senser_width/2/default_camera_lens)
    out_data = {
        'camera_angle_x': camera_angle_x,
        'camera_lens': default_camera_lens,
        'sensor_width': default_camera_senser_width,
        'env_texture': env_texture,
        'bbox_size': list(bbox_size),
        'scaling_factor': scale,
        'mesh_offset': list(mesh_offset),
        'transforms': []
    }

    parent_matrix_list = [
        rotation_matrix(0, 0, 0, 0),
    ]

    camera_locations = get_solid_points_on_sphere(
        bbox_center, distance)

    positon_tag = [convert_position(camera_location,bbox_center) for camera_location in camera_locations]

    for parent_matrix in parent_matrix_list:
        camera_idx = 0

        for camera_location in camera_locations:
            _lens = 50
            _camera_location = camera_location * \
                (_lens / default_camera_lens)
            _rotation_euler = (
                bbox_center - _camera_location).to_track_quat('-Z', 'Y').to_euler()
            cam_matrix = build_transformation_mat(
                _camera_location, _rotation_euler)
            cam_matrix = listify_matrix(parent_matrix) @ cam_matrix
            camera_params = {
                'camera_type': 'PERSP',
                'camera_lens': _lens,
                'camera_sensor_width': default_camera_senser_width,
            }
            # add_camera(cam_matrix,camera_params)
            add_camera_pose(cam_matrix, camera_params)
            index = "{0:04d}".format(idx)
            out_data['transforms'].append(listify_matrix(cam_matrix))
            idx += 1
            camera_idx += 1

    set_color_output(output_dir=output_path)
    render()

    render_opaque_flag = RENDER_DEPTH
    if render_opaque_flag:
        change_material_blend_mode()
        set_color_output(output_dir=output_path, file_prefix="render_opaque_")

    if RENDER_DEPTH:
        enable_depth_output(output_dir=output_path)
    if render_opaque_flag:
        render()

    with open(os.path.join(output_path, META_FILENAME), 'w') as out_file:
        json.dump(out_data, out_file, indent=4)

    file_prefix = "render_opaque_"
    pattern = os.path.join(output_path, f'{file_prefix}*')
    files_to_delete = glob.glob(pattern)
    for file in files_to_delete:
        os.remove(file)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print(
            "Usage: [path_to_blender] -b -P blender_render_16view.py [mesh_path] [types] [output_path]")
        exit(1)
    else:
        mesh_path = sys.argv[4]
        types = sys.argv[5]
        output_path = sys.argv[6]

        # face_dir = 'exp_results/clustering/partobjtiny/tree/00200996b8f34f55a2dd2f44d316d107_0/faces'
        # all_files = os.listdir(face_dir)
        # for f in all_files:
        #     faces = np.load(os.path.join(face_dir, f)).squeeze()
        #
        # faces = np.load(os.path.join(face_dir, "(18, np.int64(28283)).npy")).squeeze()

        ret = process(mesh_path, types, output_path)
