#!/bin/sh
#wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb
#dpkg -i cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb
#cp /var/cuda-repo-debian11-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
#DEBIAN_FRONTEND=noninteractive add-apt-repository contrib
#DEBIAN_FRONTEND=noninteractive apt-get update
#DEBIAN_FRONTEND=noninteractive apt-get -y install cuda
#wheel convert ./MultiScaleDeformableAttention-1.0-py3.9-linux-x86_64.egg
#./switch_cuda.sh 12.1
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
#export CUDA_HOME=/usr/local/cuda
#wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
#yes | ./cuda_12.1.0_530.30.02_linux.run
#pip install ./models/GroundingDINO/ops
#echo $CUDA_PATH
#echo $LD_LIBRARY_PATH
#echo $PATH
#echo $CUDA_HOME
cd ./models/GroundingDINO/ops
#python ./setup.py build install
python3 setup.py build
pip3 install .
#wheel convert ./dist/MultiScaleDeformableAttention-1.0-py3.10-linux-x86_64.egg
#mv MultiScaleDeformableAttention-1.0-py310-cp310-linux_x86_64.whl MultiScaleDeformableAttention-1.0-cp310-cp310-linux_x86_64.whl
#pip install MultiScaleDeformableAttention-1.0-cp310-cp310-linux_x86_64.whl
python ./test.py # should result in 6 lines of * True
cd ../../../
#cp ./models/GroundingDINO/ops/build/lib.linux-x86_64-cpython-310/MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so .