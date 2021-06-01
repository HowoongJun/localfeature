# EventPointNet: Supervised keypoint detector with neuromorphic camera data
This repository provides python implementation of supervised keypoint detector with neuromorphic camera data, known as EventPointNet. 

## How to run
Argument for the method is as follows

| Argument | Abbreviation | Description | Option |
|---|:---:|:---:|:---:|
|`--model`|`--m`|Model select|`eventpointnet`, `superpoint`, `sift`, `orb`|
|`--channel`|`--c`|Select channel of the image|default = `3`|
|`--mode`|`--o`|Mode Select|`makedb`, `query`, `match`, `train`|
|`--query`|`--q`|Image query file path(Only for query and match mode)||
|`--match`|`--a`|Image match file path(Only for match mode)||
|`--thresh`|`--t`|Threshold value for keypoint number|default = `3000`|
|`--db`|`--d`|DB path for training(Only for train mode)||
|`--ransac`|`--r`|RANSAC threshold value||

<br>

With the arguments above, you can run the code by executing `run.py`.
Argument mode and model are mandatory for the execution.

```bash
python run.py --mode <MODE> --model <MODEL> 
```


## Implementation

```
pytorch 1.7.1
opencv 4.5.1
```

## Training

Pre-trained model for the method is provided here (`EventPointNet/checkpoints/checkpoint.pth`)
The model was trained with [Multi Vehicle Stereo Event Camera(MVSEC) dataset](https://daniilidis-group.github.io/mvsec/). 
Left camera of the three datasets(day1, day2, and night) were used for training.

## Reference

Reference code is implemented in submodule [localfeature_ref](https://github.com/HowoongJun/localfeature_ref.git).
Details are explained in the submodule repository.

## Evaluation
