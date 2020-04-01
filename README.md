# TFRecord_TimeSeries
This repository demonstrate how to create, write and read tf.Example messages to and from .tfrecord files.

## Prerequisites
These instructions are valid for Linux Ubuntu 18.04 OS, but it should work on other OS with proper modifications.

After cloning or downloading this repository in a directory (DIR_MAIN, e.g. ~/Desktop/Projects) I strongly suggest
to install virtualenv if not already present on your machine: it will help you setting up a  project-specific python 
interpreter without installing packages system-wide.
In order to install it, execute in a terminal (Ctrl-Alt-t):
```
sudo apt-get install virtualenv      # install virtualenv
cd DIR_MAIN                          # cd to the repository directory
virtualenv venv                      # create the virtual environment
source ./venv/bin/activate           # activate it
pip install -r requirements.txt      # install the required packages
```
and you're all set. 

This repository uses the CPU version of Tenforslow 2.1.0. but the GPU accelerated package tensorflow-gpu can be
interchangeably used (the installation guide can be found [here](https://www.tensorflow.org/install/gpu)).


## TFRecords
TFrecord is an efficient way to store data leveraging [Google Protocol Buffer](https://developers.google.com/protocol-buffers/),
especially with huge datasets that don't fit in memory. 
At training time the model loads sequentially the required batches of data avoiding memory saturation. For more info please
check [tensorflow docs](https://www.tensorflow.org/tutorials/load_data/tfrecord).

## How To Use
After setting the desired set of parameters in the ```globals.py``` file, open a terminal and type:
```
cd DIR_MAIN
source ./venv/bin/activate
python main.py
```
The generated dataset will be stored in /data. 
To modify the destination folder just edit the save_dir variable in ```globals.py``` and insert the desired destination path.


