# Restarts subject to approximate sharpness

MATLAB code for the numerical experiments in the published paper [Restarts subject to approximate sharpness: a parameter-free and optimal scheme for first-order methods](https://link.springer.com/article/10.1007/s10208-024-09673-8). The preprint can also be found [here](https://arxiv.org/abs/2301.02268).

## Requirements

The experiments should run on MATLAB R2020b (or a later version) without issue.

- linspecer function ([link](https://www.mathworks.com/matlabcentral/fileexchange/42673-beautiful-and-distinguishable-line-colors-colormap)) (simply include `linspecer.m` in your MATLAB userpath)
- CVX ([link](http://cvxr.com/cvx))
- Statistics and Machine Learning Toolbox

## Running the experiments

Clone or download the repository, and set the MATLAB path to be from the repository root. 
The experiments are located in the `experiments/` folder, organized by the subsections in Section 5 of the paper.

## Attributions

The datasets in `data/libsvm-data.tar.bz2` are obtained from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). The wine data in `data/winequality.tar.bz2` is obtained from the [UCI machine learning repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

## Issues

Pertaining to the code, post questions, requests, and bugs in [Issues](https://github.com/mneyrane/restart-schemes/issues).
