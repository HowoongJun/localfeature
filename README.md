# Local feature evaluation framework
This repository provides a framework for local feature evaluation

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


## Dependencies

```
python 3.8.5
pytorch 1.7.1
opencv 4.5.2
scikit-image 0.18.1
numpy 1.19.2
```

Aside from above dependencies, docker container for this repository is provided [here](https://hub.docker.com/r/howoongjun/eventpointnet)

## Evaluation
