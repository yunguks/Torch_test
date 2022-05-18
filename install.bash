#!/bin/bash

apt-get update -y
apt-get upgrade -y
apt-get install vim -y

#apt-get install ssh -y
#apt-get install openssh-server -y
#service ssh start
#apt-get install python3-opencv -y

pip install -r requirments.txt
#apt-get install jupyter-notebook -y