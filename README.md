# RPG MAP GAN
This app creates maps for RPG games in the style of Divinity: Original Sin II minimaps, using generative adversarial networks. It includes a range of tools that can be used to create additional datasets, as well as code necesary for training new models.

## Data_utils
### gui_app.py
A desktop app, used to transform images to arrays of labels representing various terrain types.
### dataSetCreator.py
A script for creating datasets from data processed by gui_app.

## Models
Contains two models - discriminator and generator, as well as a script for training them.

## User_app
Contains the actual app. The models are too large to put on github, so here's a link to Google Drive:
https://drive.google.com/drive/folders/1JNMyVcvEqSahotVdQGqa8KbcrPr36erj?usp=sharing