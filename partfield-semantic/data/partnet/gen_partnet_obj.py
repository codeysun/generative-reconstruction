import numpy as np
import os
import argparse
import json
from collections import defaultdict

def read_selected_directories(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)

    return v, f

def export_obj(out, v, f):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

def gen_data(partnet_dir, exp, output_dir):
    """create data format (train images and transforms, gt seg(later))"""

    cur_shape_dir = os.path.join(partnet_dir, exp)
    cur_part_dir = os.path.join(cur_shape_dir, 'objs')
    leaf_part_ids = [item.split('.')[0] for item in os.listdir(cur_part_dir) if item.endswith('.obj')]


    # Load objs
    root_v_list = []; root_f_list = []; tot_v_num = 0;
    obj_to_face_start_index = {} # store starting index of the faces from each obj
    face_count = 0
    obj_counts = {} #store count of faces from each obj
    for idx in leaf_part_ids:
        v, f = load_obj(os.path.join(cur_part_dir, str(idx)+'.obj'))
        mesh = dict();
        mesh['v'] = v; mesh['f'] = f;
        root_v_list.append(v);
        root_f_list.append(f+tot_v_num);
        tot_v_num += v.shape[0];
        obj_to_face_start_index[idx] = face_count
        face_count += len(f)
        obj_counts[idx] = len(f)

    root_v = np.vstack(root_v_list)
    root_f = np.vstack(root_f_list)

    center = np.mean(root_v, axis=0)
    root_v -= center
    scale = np.sqrt(np.max(np.sum(root_v**2, axis=1)))
    root_v /= scale

    mesh_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_file = os.path.join(mesh_dir, f"{exp}.obj")
    export_obj(mesh_file, root_v, root_f)
    print(f"Combined mesh saved to: {mesh_file}")

    # Create ground truth semantic file
    result_json = os.path.join(cur_shape_dir, 'result.json')
    meta_json = os.path.join(cur_shape_dir, 'meta.json')
    with open(result_json, 'r') as fin:
        tree_hier = json.load(fin)[0]
    with open(meta_json, 'r') as fin:
        meta = json.load(fin)
    category = meta['model_cat']

    semantic_face_map = {}
    def traverse_hierarchy(node, current_labels=[]):
        current_labels = current_labels + [node["text"]]
        if 'objs' in node.keys():  # Leaf node (OBJ files)
            for obj_id in node['objs']:
                if obj_id not in obj_to_face_start_index:
                    print (f"Warning {obj_id} not in obj_to_face_start_index")
                    continue
                start_index = obj_to_face_start_index[obj_id]
                count = obj_counts[obj_id]
                face_indices = list(range(start_index, start_index + count))
                for label in current_labels:
                    if label not in semantic_face_map:
                        semantic_face_map[label] = []
                    semantic_face_map[label].extend(face_indices)

        if 'children' in node:
            for child in node['children']:
                traverse_hierarchy(child, current_labels)

    traverse_hierarchy(tree_hier)

    label_list = list(semantic_face_map.keys())
    num_labels = len(label_list)
    num_faces = len(root_f)
    face_matrix = np.zeros((num_labels, num_faces), dtype=np.uint8) # Use uint8 for memory efficiency
    for i, label in enumerate(label_list):
        face_indices = semantic_face_map[label]
        face_matrix[i, face_indices] = 1  # Set corresponding indices to 1

    # save the face matrix
    gt_sem_dir = os.path.join(output_dir, "semantic_gt")
    os.makedirs(gt_sem_dir, exist_ok=True)
    gt_sem_path = os.path.join(gt_sem_dir, f"{exp}.npy")
    np.save(gt_sem_path, face_matrix)

    return category, label_list


def main(args):
    def read_selected_directories(file_path):
        with open(file_path, "r") as f:
            return [line.strip() for line in f]

    shape_list_txt = 'selected_partnet_data_all_categories.txt'
    shape_list = set(read_selected_directories(shape_list_txt))
    shape_list_extra = [10558, 8677, 555, 14102, 11526, 7289, 23894, 10101, 1135, 1128, 10806]

    for shape in shape_list_extra:
        shape_list.add(str(shape))

    SHAPE_LIST = list(shape_list)
    # SHAPE_LIST = shape_list_extra

    # Create gt json file
    output_dir = "data/partnet"
    os.makedirs(output_dir, exist_ok=True)
    gt_data = defaultdict(dict)

    for shape in SHAPE_LIST:
        category, labels = gen_data(args.partnet_dir, str(shape), output_dir)
        gt_data[category][shape] = labels

    with open(os.path.join(output_dir, "partnet_semantic.json"), 'w') as f:
        json.dump(gt_data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # PartNet file
    parser.add_argument('--partnet_dir', type=str, default='/home/codeysun/git/data/PartNet/data_v0/')
    args = parser.parse_args()

    main(args)
