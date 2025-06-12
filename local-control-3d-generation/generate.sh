#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./extern/MVDream

# Define your lists
prompts=(
  "A bowl of bananas"
  # "A small statue of an angel, 3D asset"
  # "A children's toy sailboat, 3D asset"
  # "An Oscar academy award trophy, 3D asset"
  # "A military tank, 3D asset"
  # "A statue figurine of a dancing ballerina on one leg, 3D asset"
  # "A high-tech cyber headset, 3D asset"
  # "An extravagant diamond engagement ring, 3D asset"
  # "A high-resolution rendering of a fox, 3D asset"
  # "A high-resolution rendering of a rustic coffee table, 3D asset"
  # "A chair with wings on the back, 3D asset"
  # "A luxury designer leather handbag with gold accents, 3D asset"
  # "A flat L-shape style black faucet, 3D asset"
)

# Define your file paths list
file_paths=(
  "/home/codeysun/git/gen-recon/partfield-semantic/data/objaverse_samples/0c3ca2b32545416f8f1e6f0e87def1a6.glb"
  # "/home/codeysun/git/data/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/670f26a742f748b6b7542e7730f9803a.glb"
  # "/home/codeysun/git/data/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/00aee5c2fef743d69421bb642d446a5b.glb"
  # "/home/codeysun/git/data/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/2dd2980caf87450d9ef7e42adb43f2ca.glb"
  # "/home/codeysun/git/data/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/2dd2980caf87450d9ef7e42adb43f2ca.glb"
  # "/home/codeysun/git/data/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/9d6b66aba431493097f289a6e42f3614.glb"
  # "/home/codeysun/git/data/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/c93ee36deda14b4aa73c2b5a9d9e9c9f.glb"
  # "/home/codeysun/git/data/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/074c0b05d9644dd0ab5e2b932fa2bb5a.glb"
  # "/home/codeysun/git/gen-recon/partfield-semantic/data/partnet/mesh/8677.obj"
  # "/home/codeysun/git/gen-recon/partfield-semantic/data/partnet/mesh/25256.obj"
  # "/home/codeysun/git/gen-recon/partfield-semantic/data/partnet/mesh/555.obj"
  # "/home/codeysun/git/gen-recon/partfield-semantic/data/partnet/mesh/13216.obj"
  # "/home/codeysun/git/gen-recon/partfield-semantic/data/partnet/mesh/2094.obj"
)

# Define your masked segments list
masked_segments=(
  "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/objaverse/sem_preds/0c3ca2b32545416f8f1e6f0e87def1a6_0/bowl_1.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partobjtiny/sem_preds/670f26a742f748b6b7542e7730f9803a/Body.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partobjtiny/sem_preds/00aee5c2fef743d69421bb642d446a5b/Basket.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partobjtiny/sem_preds/2dd2980caf87450d9ef7e42adb43f2ca/Base.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partobjtiny/sem_preds/2dd2980caf87450d9ef7e42adb43f2ca/Rolleres.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partobjtiny/sem_preds/9d6b66aba431493097f289a6e42f3614/Mushroom Body.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partobjtiny/sem_preds/c93ee36deda14b4aa73c2b5a9d9e9c9f/Earphone.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partobjtiny/sem_preds/074c0b05d9644dd0ab5e2b932fa2bb5a/Watchband.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partnet/sem_preds/8677/Handle.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partnet/sem_preds/25256/Tabletop.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partnet/sem_preds/555/Seat Surface.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partnet/sem_preds/13216/Handle.npy"
  # "/home/codeysun/git/gen-recon/partfield-semantic/exp_results/clustering/partnet/sem_preds/2094/Vertical Support.npy"
)

# Ensure lists are of the same length
if [[ ${#prompts[@]} -ne ${#file_paths[@]} || ${#prompts[@]} -ne ${#masked_segments[@]} ]]; then
  echo "Error: The lists must be of the same length."
  exit 1
fi

# Loop through each combination of prompt, file_path, and masked_segment
for ((i=0; i<${#prompts[@]}; i++)); do
  prompt=${prompts[i]}
  file_path=${file_paths[i]}
  masked_segment=${masked_segments[i]}

  # Run the Python script with the current parameters
  python launch.py --config configs/controldreamernerfstrict-sd21-shading.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$prompt" \
    system.control_renderer.file_path="$file_path" \
    system.control_renderer.masked_segments="$masked_segment" \
    system.control_renderer.local_control=True \
    system.guidance.control_scale=2.0

  # python launch.py --config configs/controldreamernerfstrict-sd21-shading.yaml --train --gpu 0 \
  #   system.prompt_processor.prompt="$prompt" \
  #   system.control_renderer.file_path="$file_path" \
  #   system.control_renderer.masked_segments="$masked_segment" \
  #   system.control_renderer.local_control=False \
  #   system.guidance.control_scale=1.0

done
