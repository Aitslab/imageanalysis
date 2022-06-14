#Project code
This folder contains all the code for the project

The **Hover_net_versions** folder has two subfolders for our work on the tensorflow version of the HoVer net architecture, as well as the pytorch version of it

**HoverNet_evaluation_nuclei.ipynb** is a jupyter notebook containing code for evaluating our results on an instance based manner, as distinct from a pixel based one. The code is based on Malou Arvidssons work in the github folder https://github.com/Aitslab/imageanalysis/tree/master/Malou_dir/Results/Model-Evaluations 

**Hovernet_epochs_plots.ipynb** contains evaluation of the results for each epoch with a pixel based metrics

**MicroNet_reimplementation.ipynb** is code based on the https://github.com/ZeyuGaoAi/Instance_based_Vision_Transformer repository, running inference on the micronet archtecture. This code has **NOT** not contributed to the final reusults, but is included to show another attempt we did along when experimenting between the micronet and the HoVernet architecture.

**example_code.ipynb** is the HoVer net archtecture code from the https://github.com/vqdang/hover_net/blob/tensorflow-final repo with comments to understand each methods connection to the architecture (example picture in the the diagram.png image a level up in the folder structure)

**preprocess_images.m** is a matlab script for converting annotations in the format of an image to an .mat file, with the nessesary information from all the nuclei. It also contains code to convert a gray scale image to one in three channel RGB format.



