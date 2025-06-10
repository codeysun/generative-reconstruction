from __future__ import annotations
import os
import imageio
import pickle
import numpy as np
import json
from graphviz import Digraph
from PIL import Image, ImageOps

from renderer import NVDiffRasterizer
from partfield.utils import gen_camera_traj

class TreeNode:
    def __init__(self, label, mesh=None, image=None, image_path=None, parent=None) -> None:
        self.label = label

        # TODO: change to better name. This is really the face indices of the submesh
        self.mesh = mesh

        # self.set_image(image, image_path)

        self.children = []
        self.parent = parent if parent is not None else self

        self.images = None
        self.image_paths = None
        self.query_images = None
        self.query_image_paths = None
        self.caption = ''

    def add_child(self, child_node : TreeNode) -> None:
        self.children.append(child_node)

    def set_caption(self, caption) -> None:
        self.caption = caption

    def set_image(self, image, image_path) -> None:
        if image is not None and not isinstance(image, list):
            self.images = [image]
        else:
            self.images = image or []

        if image_path is not None and not isinstance(image_path, list):
            self.image_paths = [os.path.abspath( image_path )]
        else:
            self.image_paths = [os.path.abspath(p) for p in image_path] or []

    def set_mesh(self, mesh) -> None:
        self.mesh = mesh

    def set_embed(self, embed) -> None:
        self.embed = embed

class PartTree:
    def __init__(self, label, mesh=None, image=None, image_path=None) -> None:

        face_indices = np.indices(mesh.faces.shape[:-1]).squeeze()
        self.root = TreeNode(label, mesh=face_indices, image=image, image_path=image_path)
        self.root_mesh = mesh
        self.label_to_node = {label: self.root}

    def get_nodes(self) -> list:
        return list(self.label_to_node.values())
    
    def get_leaves(self) ->list:
        node_list = self.get_nodes()
        return [node for node in node_list if not node.children]

    def set_render_dir(self, render_dir)-> None:
        self.render_dir = render_dir

    def set_node_mesh(self, label, mesh) -> None:
        assert self.exists(label), f"Node {label} does not exist"
        node = self.label_to_node[label]
        node.set_mesh(mesh)

    def set_node_image(self, label, image, image_path) -> None:
        assert self.exists(label), f"Node {label} does not exist"
        node = self.label_to_node[label]
        node.set_image(image, image_path)

    def set_node_embed(self, label, embed) -> None:
        assert self.exists(label), f"Node {label} does not exist"
        node = self.label_to_node[label]
        node.set_embed(embed)

    def add_edge(self, child_label, mesh=None, image=None, image_path=None, parent_label=None) -> None:
        if parent_label is None:
            parent = self.root
        else:
            if parent_label not in self.label_to_node:
                raise KeyError(f"Label {parent_label} not in tree")
            parent = self.label_to_node[parent_label]

        if self.exists(child_label):
            raise ValueError(f"Label of {child_label} already exists.")
        child_node = TreeNode(child_label, mesh, image, image_path, parent=parent)
        parent.add_child(child_node)
        self.label_to_node[child_label] = child_node

    def exists(self, label) -> bool:
        return label in self.label_to_node

    def canonize_tree(self) -> None:
        """
        Consolidate nodes with single children
        """
        def canonize(node):
            while len(node.children) == 1:
                self.label_to_node.pop(node.children[0].label)
                node.children = node.children[0].children
            for child in node.children:
                child.parent = node
                canonize(child)
        canonize(self.root)

    def generate_images_from_mesh(self, output_dir, num_views=16, img_size=512, blender=False) -> None:
        rast = NVDiffRasterizer(self.root_mesh, color=[0.5, 0.5, 0.5], img_size=img_size)

        # Generate camera trajectory
        c2ws = gen_camera_traj(num_views, blender=blender)

        def gen_images(node, c2ws, rast, output_dir):
            # Render the obj along camera trajectory
            image_paths = []
            rast.set_mesh(node.mesh)

            for i, c2w in enumerate(c2ws):
                c2w_batched = c2w[None, ...] # add batch dim
                control_dict = rast(c2w_batched)

                rgb = control_dict["comp_rgb"].squeeze().detach().cpu().numpy() # H W C
                mask = control_dict["mask"].squeeze().detach().cpu().numpy() # H W
                # mask[mask < 1.0] = 0.0
                mask_bool = mask >= 1.0 - 1e-6

                # if node is not self.root:
                rgb[~mask_bool] = [0, 0, 0] # black
                # rgb[~mask_bool] = [1.0, 1.0, 1.0] # white bg

                image = (rgb * 255).astype(np.uint8)
                alpha = (mask * 255).astype(np.uint8)
                # convert to RGBA
                image = np.concatenate((image, alpha[..., None]), axis=2)


                image = Image.fromarray(image)

                img_dir = os.path.join(output_dir, 'masks', str(node.label))
                os.makedirs(img_dir, exist_ok=True)
                image_path = os.path.join(img_dir, f"{i:04d}.png")
                image.save(image_path)
                image_paths.append(image_path)

            node.set_image(image=None, image_path=image_paths)

            for child in node.children:
                gen_images(child, c2ws, rast, output_dir)


        gen_images(self.root, c2ws, rast, output_dir)

    def save_images(self) -> None:
        assert self.root.images, "No images in the tree"

        def save_node_images(node: TreeNode):
            for i in range(len(node.images)):
                if not os.path.exists(node.image_paths[i]):
                    os.makedirs(os.path.dirname(node.image_paths[i]), exist_ok=True)
                imageio.imsave(node.image_paths[i], node.images[i])
                if node.query_images:
                    imageio.imsave(node.query_image_paths[i], node.query_images[i])

            for child in node.children:
                save_node_images(child)

        save_node_images(self.root)


    def add_graphviz_edge(self, node: TreeNode, graph, idx=0):
        # Use HTML-like label to embed images in nodes
        if node.query_image_paths:
            im = node.query_image_paths[idx]
        else:
            im = node.image_paths[idx]

        graph.node(str(node.label), label=node.caption, image=im, 
                   imagepos='tc', labelloc='b', shape='plaintext', fontcolor='green')
        for child in node.children:
            graph.edge(str(node.label), str(child.label))
            self.add_graphviz_edge(child, graph, idx)

    def render_tree(self, output_file) -> None:
        """
        Render the entire part tree using the provided images
        """
        os.makedirs(output_file, exist_ok=True)

        # self.save_images()

        img_paths = self.root.query_image_paths or self.root.image_paths

        for idx in range(len(img_paths)):
            dot = Digraph(comment='Tree with Images', format='pdf', engine='dot')
            dot.attr(size='100,100')  # Size can be adjusted
            self.add_graphviz_edge(self.root, dot, idx)
            dot.render(os.path.join(output_file, f"{idx:03d}"), view=False)  # This creates a PNG and opens it

    def query_preprocess(self, **kwargs) -> None:
        """
        Create query images to be fed to image encoder
        Query images consist of the parent image, with the query segment masked
        with a blue overlay
        """
        assert self.root.image_paths, "No image paths in this tree"
        assert self.render_dir, "No render dir specified for canonical views"
        num_images = len(self.root.image_paths)
        parent_imgs = []
        for i in range(num_images):
            # image = Image.open(os.path.join(self.render_dir, f"render_{i:04d}.webp"))
            image = Image.open(self.root.image_paths[i])
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            parent_imgs.append(image)

        def make_query_image(node: TreeNode, parent_imgs, **kwargs) -> None:
            query_images = []
            query_image_paths = []

            assert node.image_paths, "No image paths in this tree"
            for i in range(len(node.image_paths)):
                if node.images:
                    node_img = Image.fromarray(node.images[i])
                    mask_img = node_img.convert("L")
                    mask_img = mask_img.point(lambda x : 128 if x > 0 else 0, 'L')
                elif node.image_paths:
                    node_img = np.array(Image.open(node.image_paths[i]))
                    if node_img.shape[-1] == 4:
                        mask_img = node_img[..., 3] != 0
                        node_img[~mask_img] = [255, 255, 255, 255]
                        node_img = node_img[..., :3]
                        mask_img = Image.fromarray(mask_img)
                        mask_img = mask_img.point(lambda x : 128 if x > 0 else 0, 'L')
                    node_img = Image.fromarray(node_img)
                else:
                    raise Exception("No image or image paths in this tree")

                query_img = node_img

                if 'crop' in kwargs:
                    bbox = node_img.getbbox()
                    query_img = query_img.crop(bbox)

                if 'pad' in kwargs:
                    h, w = query_img.size
                    l = max(w,h)
                    size = (l, l)
                    query_img = ImageOps.pad(query_img, size, color=(0, 0, 0))

                if 'resize' in kwargs:
                    size = kwargs['resize']
                    query_img = query_img.resize(size)

                if 'highlight' in kwargs:
                    # Query image will be the root image with region highlighted
                    # node_img = node_img.convert("RGBA")
                    if node is not self.root:
                        parent_img = parent_imgs[i] # RGBA

                        overlay = Image.new("RGBA", node_img.size, (255, 0, 0, 255))
                        query_img = Image.composite(overlay, parent_img, mask_img)
                        query_img.putalpha(parent_img.getchannel('A'))

                # Convert to RGB with white background
                query_img = np.array(query_img)
                if query_img.shape[-1] == 4:
                    mask = query_img[..., 3] != 0
                    query_img[~mask] = [255, 255, 255, 255]
                    query_img = query_img[..., :3]
                query_img = Image.fromarray(query_img.astype('uint8'))

                query_img_path = node.image_paths[i][:-4] + '_query.png'
                query_img.save(query_img_path)
                # query_img = np.array(query_img)
                # query_images.append(query_img)
                query_image_paths.append(query_img_path)

            # node.query_images = query_images
            node.query_image_paths = query_image_paths

            for child in node.children:
                make_query_image(child, parent_imgs, **kwargs)


        make_query_image(self.root, parent_imgs, **kwargs)
        
    def format_json(self):
        """
        Format the tree as a json
        """
        def tree_to_dict(node):
            node_dict = {"label": node.label,
                         "image": node.query_image}
            if node.children:
                node_dict["children"] = []
                for child in node.children:
                    node_dict["children"].append(tree_to_dict(child))
            return node_dict
        
        tree_dict = tree_to_dict(self.root)
        return tree_dict

    def save_face_indices(self, output_dir) -> None:
        os.makedirs(output_dir, exist_ok=True)
        def save_faces(node):
            fname = os.path.join(output_dir, str(node.label))
            np.save(fname, node.mesh)

            for child in node.children:
                save_faces(child)

        save_faces(self.root)



def save_tree(tree, output_path) -> None:
    with open(output_path, 'wb') as f:
        pickle.dump(tree, f)
    print(f'Tree saved to {output_path}')

def load_tree(path) -> PartTree:
    with open(path, 'rb') as f:
        tree = pickle.load(f)
    return tree



