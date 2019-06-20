# Navigation Model for OpenStreetSet (OSS)

### Dependencies

* [Pytorch](https://pytorch.org/) - Machine learning library for Python-related dependencies
* [Anaconda](https://www.anaconda.com/download/) - Anaconda includes all the other Python-related dependencies
* [ArgParse](https://docs.python.org/3/library/argparse.html) - Command line parsing in Python
 
### Instructions
 - Here are the instructions to use the code base:
 
##### Train Model:
 - To train the model with options, use the command line:
```sh
$ python train_model.py --options %(For the details of options)
$ python train_model.py [-h] [short_name_arg] %(For explanation on the commands)
```
 - An example of running:
 ```sh
$ python train_model.py -m1 'map_1' -m2 'map_2' -me  30 -do 0.9
```
##### Test Model:
 - Choose a model to evaluate on test map, with the command line:
```sh
$ python test_model.py --options %(For the details of options)
$ python test_model.py [-h] [short_name_arg] %(For explanation on the commands)
```

