# nuke_kaggle
## Data Science Bowl Nucleus Project

`git clone https://github.com/jinkos/nuke_kaggle`

will create a nuke_kaggle directory with all the files in it.

edit the config file and add and entry for your machine_name

`python3 image_wrapper.py your_machine_name`

Create the aug files. 

Used to take 10 hours but I have added multiprocessing. My LINUX box manages it in 2 hours, now. I had to use pillow to save the .png files because skimage wasn't working. So you need to install pillow

`python3 train.py your_machine_name --save`

should train the model and save the weights

`python3 train.py your_machine_name --load --save`

will load the weights and then update them

python3 submit.py your_machine_name will enable you to analyse your results or make a submission depending on how you have set the flags in the code.


