# General restart schemes

MATLAB code for restart schemes, *currently a work-in-progress!*

If you plan to run anything here, you should know that

- the radial restart scheme code was updated, now the NESTA and FISTA examples do not work

## Requirements

Runs on MATLAB R2020b without issue. *TO DO: list toolboxes if any are used.*

- linspecer function ([link](https://www.mathworks.com/matlabcentral/fileexchange/42673-beautiful-and-distinguishable-line-colors-colormap)) (simply include `linspecer.m` in your MATLAB userpath)
- CVX ([link](http://cvxr.com/cvx))
- Statistics and Machine Learning Toolbox

## Running the experiments

Clone or download the repository, and set the MATLAB path to be from the repository root. 
The experiments are the `ne_*.m` scripts.

## Attributions

The datasets in `data/libsvm-data.tar.bz2` are obtained from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). The wine data in `data/winequality.tar.bz2` is obtained from the [UCI machine learning repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

## Issues

Pertaining to the code, post questions, requests, and bugs in [Issues](https://github.com/mneyrane/restart-schemes/issues).
