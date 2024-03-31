# FS-Depth-v1
The pytorch implementation of the paper FS-Depth.

Our base model ZoeDepth:https://github.com/isl-org/ZoeDepth


# training
python train_v1.py -m zoedepth --pretrained_resource=""

# Evaluating
python evaluate.py -m zoedepth --pretrained_resource="local::/path/to/local/ckpt.pt" -d dataset name
