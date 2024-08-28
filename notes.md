## instance start
- interactive 

	`qsub -I -q gpu -l walltime=8:00:00 -l select=1:ncpus=1:ngpus=1:gpu_mem=20000mb:scratch_local=10gb:gpu_cap=cuda75`

	`qsub -I -q iti -l walltime=8:00:00 -l select=1:ncpus=1:ngpus=1:gpu_mem=8000mb:scratch_local=10gb`

## tmux
- ctrl + b + [something]

## build docker
in mmdetection-2.28...
docker build -t mmdet2 docker/

## loading the environment
### modules

	module avail

	module load conda-modules

- cuda? 

## singularity 

### locally docker -> save imge
1. docker save [image] -o image.tar
2. upload image.tar (if building remotely)

### build singularity
1. module load singularity
2. singularity build --sandbox mmdet2 docker-archive://image.tar
2. singulairyt build mmdet.sif docker-archive://image.tar
3. singularity shell --nv  mmdet2


sudo singularity build name.sif 

#PBS -N fastRcnnTrafficSigns1
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=20000mb:scratch_local=10gb
#PBS -l walltime=8:00:00 
#PBS -m ae
