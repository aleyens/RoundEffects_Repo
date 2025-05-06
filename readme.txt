This repository contains the replication package for the paper "Round effects in economic experiments - A novel probabilistic programming approach to round-dependent response behaviour" by Alexa Leyens, Philipp Feisthauer, Jan Börner, Monika Hartmann and Hugo Storm.

The main data file used for modeling is accessible online via https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/979019ef-037a-4716-9c40-dfa0125f16f4 and can be used without prior downloading when running the model code provided in this repository. 

In order to replicate the figures and analyses conducted in the paper, several software requirements need to be fulfilled. To ease the replication, we have created a docker container that anyone can replicate to enable their local machines to run our modeling code. 
For setting up a docker installation on your machine, follow the steps described in: https://docs.docker.com/get-started/introduction/get-docker-desktop/

Then, to replicate the analyses and figures in the paper, follow these steps:
1. Pull the repository from GitHub: git clone https://github.com/aleyens/RoundEffects_Repo 
2. Rebuild docker image locally by running: docker build 
3. Start the container from this image: docker run 

Once the container is running, you can attach a shell and run the code provided in the 'model...'-file.

The Python requirements are listed in requirements.txt and can be manually installed if the replicator does not want to use the docker container. 

