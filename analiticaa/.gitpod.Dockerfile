FROM gitpod/workspace-full:latest 
RUN sudo apt-get update
RUN sudo apt-get install -y python3-venv python3-pip
USER gitpod