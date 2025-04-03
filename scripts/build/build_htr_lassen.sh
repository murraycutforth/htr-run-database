module purge
module load gcc/7.3.1
module load cuda/11.1.1
module load cmake/3.14.5
module load python/3.8.2

# Build config
export CC=gcc
export CXX=g++
export CONDUIT=ibv

# CUDA config
export CUDA_HOME=/usr/tce/packages/cuda/cuda-11.1.0
export CUDA="$CUDA_HOME"
export GPU_ARCH=volta
	
# Path setup
export LEGION_DIR=/usr/workspace/cutforth1/legion_lassen
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=/usr/workspace/cutforth1/htr_lassen
export SCRATCH=/p/gpfs1/cutforth1/
export GROUP="stanford"
	
# Legion setup
export USE_CUDA=1
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
#export GASNET_VERSION="GASNet-1.32.0"
	
# HTR Preferences
export DEBUG=0
export CURVILINEAR_COORDINATES=1
export RESERVED_CORES=8
export NO_ATOMIC=0

git clone https://gitlab.com/insieme1/htr/htr-solver.git htr_lassen
cd htr_lassen
git checkout 13be1790fcaccb25d34138b8f79799a71cf588a4
cd src

echo "Printing current modules:"
module list
echo "Starting legion compilation..."

make -j prometeo_BFERoxy6SpMix.exec
