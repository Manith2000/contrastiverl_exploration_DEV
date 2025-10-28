#!/bin/bash

# do NOT load the old system CUDA. JAX 12.x bundles its own libraries.
# module load cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0

pip install optax==0.1.7
pip install --upgrade jax==0.4.7

# install the jaxlib version for CUDA 12
pip install --upgrade jaxlib==0.4.7+cuda12.cudnn88 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# no longer need to set the library path
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/{path to cuda}/lib64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# #!/bin/bash

# echo "Loading modules..."
# #module load cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0

# echo "Installing Python packages..."
# pip install optax==0.1.7
# pip install --upgrade jax==0.4.7 jaxlib==0.4.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# echo "Setting library paths..."
# # Use the $CUDA_HOME variable set by the 'module load' command
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# echo "Setup complete."