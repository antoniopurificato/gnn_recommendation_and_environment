# Eco-Aware Graph Neural Networks for Sustainable Recommendations

## Setup

- Clone the repo:

``` git clone https://github.com/antoniopurificato/gnn_recommendation_and_environment.git && cd gnn_recommendation_and_environment-98E7```

- Install Pytorch Geometric:

```pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html```

- Install the required packages:

```pip install -r requirements.txt```

- Select as current directory Recbole-GNN:

```cd Recbole-GNN```

- Run an experiment:

```python3 train.py --models [LightGCN, NGCF, SimGCL, LightGCL] --datasets [ml-1m, diangping, amazon-beauty] --epochs NUM_EPOCHS --embeddings_sizes EMB_SIZES```

An example:

```python3 train.py --models NGCF --datasets ml-1m --epochs 10 --embeddings_sizes 32```

Remember than you can put in the terminal multiple values for the epochs and the embedding sizes and the code will automatically sweep on that parameters.

Sometimes it happens that DianPing generates an error with the RecBole library. If this is the case use the following command:

```mv dataset/dianping/DianPing/DianPing.inter dataset/dianping/dianping.inter && mv dataset/dianping/DianPing/DianPing.item dataset/dianping/dianing.item```

## Acknowledgments:

This project is based on the RecBole library:

```bibtex
@inproceedings{zhao2022recbole2,
  author={Wayne Xin Zhao and Yupeng Hou and Xingyu Pan and Chen Yang and Zeyu Zhang and Zihan Lin and Jingsen Zhang and Shuqing Bian and Jiakai Tang and Wenqi Sun and Yushuo Chen and Lanling Xu and Gaowei Zhang and Zhen Tian and Changxin Tian and Shanlei Mu and Xinyan Fan and Xu Chen and Ji-Rong Wen},
  title={RecBole 2.0: Towards a More Up-to-Date Recommendation Library},
  booktitle = {{CIKM}},
  year={2022}
}

@inproceedings{zhao2021recbole,
  author    = {Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Yushuo Chen and Xingyu Pan and Kaiyuan Li and Yujie Lu and Hui Wang and Changxin Tian and  Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji{-}Rong Wen},
  title     = {RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
  booktitle = {{CIKM}},
  pages     = {4653--4664},
  publisher = {{ACM}},
  year      = {2021}
}
```
