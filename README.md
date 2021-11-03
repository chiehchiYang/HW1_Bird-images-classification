# Bird images classification
## HW1 of Selected Topics in virsual Recognition using Deep Learning

## Getting started

Use git clone to download the files

First, create a new vm using conda
```shell
conda create -n hw1 python=3.6
conda activate hw1
```

Second, install the development requirements
```shell
pip install -r requirements.txt
```

## Train one epoch

use command line to create the folder(model_save)

```shell
cd HW1_Bird-images-classification 
mkdir model_save
python3 train.py
```

The result will save all the model to folder model_save


## Test the model
Please download the model from below link:
https://drive.google.com/file/d/1wXYdOefjaLRo8QfEn1a2xKvVdGIYNE8O/view?usp=sharing

```shell
python3 val.py
```
use val.ipynb to create the answer.txt


