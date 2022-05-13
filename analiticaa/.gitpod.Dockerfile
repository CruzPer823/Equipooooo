FROM gitpod/workspace-base 
RUN sudo apt-get update
RUN sudo apt-get install -y python3-venv python3-pip
USER gitpod
RUN pip install pandas seaborn scikit-learn kneed