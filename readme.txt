readme

This repository contains the replication package for the paper "Round effects in economic experiments - A novel probabilistic programming approach to round-dependent response behaviour" by Alexa Leyens, Philipp Feisthauer, Jan Börner, Monika Hartmann and Hugo Storm.

The main data file used for modeling is accessible online via https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/979019ef-037a-4716-9c40-dfa0125f16f4 and can be used without prior downloading when running the model code provided in this repository. 

In order to replicate the figures and analyses conducted in the paper, follow these steps: 
1. Pull the repository from GitHub: git clone https://github.com/aleyens/RoundEffects_Repo 
2. Rebuild docker image locally by running: docker build 
3. Start the container from this image: docker run 

Once the container is running, you can attach a shell and run the code provided in the 'model...'-file.

Software requirements are listed in the Dockerfile. Python requirements are listed in requirements.txt. When using the provided docker image from Docker Hub, all software requirements are installed.

