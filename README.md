## Background

The retrieval stage plays a critical role in advertisement recommendation systems. In industrial practice, multi-channel merging is widely adopted to select candidate ads from multiple retrieval channels. A key yet underexplored challenge is how many ads each channel should contribute to maximize overall performance.

We address this problem by modeling multi-channel merging as a cooperative game and using Shapley value–based allocation to evaluate channel contributions and design optimal retrieval strategies.

## How to use the code

### Set up and activate the environment

- For mac: conda env create -f environment.yml
- For win: conda env create -f environment_win.yml
- conda activate merged-env

### Start main
- cd /your/path/to/0_channel (e.g. /Users/denghaonan/Desktop/AuctionNet-main/0_channel)
- python 0_shapely_single.py

