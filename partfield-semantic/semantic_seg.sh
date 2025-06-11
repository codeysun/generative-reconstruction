#!/bin/bash

# Output config
FEATURE_DIR="partfield_features/objaverse"
OUT_DIR="exp_results/clustering/objaverse"

# Data config
DATA_PATH="data/objaverse_samples"
# GT_PATH="data/objaverse_samples/semantic.json"

# Environment config
CONFIG_PATH="configs/final/demo.yaml"
CONTINUE_CKPT="model/model_objaverse.ckpt"
BLENDER_PATH="blender-4.0.0-linux-x64/blender"

# First, run PartField inferencing.
pip install -U torch==2.4 torchvision torchaudio
python partfield_inference.py -c $CONFIG_PATH --opts continue_ckpt $CONTINUE_CKPT result_name $FEATURE_DIR dataset.data_path $DATA_PATH

# Then, cluster and build a hierarchical segment tree.
python tree_vis.py --root exp_results/$FEATURE_DIR --dump_dir $OUT_DIR --source_dir $DATA_PATH --use_agglo True --max_num_clusters 20 --option 0

# Render 360 degree views with Blender (if using .glb meshes).
python tree_render_views.py --blender_path $BLENDER_PATH --out_dir $OUT_DIR --source_dir $DATA_PATH

# Perform part labeling (ensure PyTorch is upgraded to 2.6).
pip install -U torch==2.6 torchvision torchaudio
python tree_query.py --out_dir $OUT_DIR --label_trees
