# nuke_kaggle
## Data Science Bowl Nucleus Project

`git clone https://github.com/jinkos/nuke_kaggle`

will create a nuke_kaggle directory with all the files in it.

edit the config file and add and entry for your machine_name

`python3 image_wrapper.py your_machine_name`

should create the aug files. Take a couple of hours. I have added multiprocessing so it should be about 4 * faster if you have 4 CPUs

`python3 train.py your_machine_name --save`

should train the model and save the weights

`python3 train.py your_machine_name --load --save`

will load the weights and then update them

python3 submit.py your_machine_name will enable you to analyse your results or make a submission depending on how you have set the flags in the code.


