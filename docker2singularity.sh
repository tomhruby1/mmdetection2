#!/bin/bash
# build singulairy image  - sif file based on input docker image
# use: build_singularity docker_image_id singularity_name

rm -rf singularity
mkdir singularity 

echo "saving docker image..."
docker save $1 -o singularity/image.tar
echo "building singularity image"
singularity build singularity/$2.sif docker-archive://singularity/image.tar 
singularity shell --nv singularity/$2.sif
