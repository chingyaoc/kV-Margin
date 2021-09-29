# Measuring Generalization with Optimal Transport

<p align='left'>
<img src='https://github.com/chingyaoc/kV-Margin/blob/main/fig.png?raw=true' width='700'/>
</p>

Understanding the generalization of deep neural networks is one of the most important tasks in deep learning. Although much progress has been made, theoretical error bounds still often behave disparately from empirical observations. In this work, we develop margin-based generalization bounds, where the margins are normalized with optimal transport costs between independent random subsets sampled from the training distribution. In particular, the optimal transport cost can be interpreted as a generalization of variance which captures the structural properties of the learned feature space. Our bounds robustly predict the generalization error, given training data and network parameters, on large scale datasets. Theoretically, we demonstrate that the concentration and separation of features play crucial roles in generalization, supporting empirical results in the literature.


**Measuring Generalization with Optimal Transport** NeurIPS 2021 [[paper]](https://arxiv.org/abs/2106.03314)
<br/>
[Ching-Yao Chuang](https://chingyaoc.github.io/), 
[Youssef Mroueh](https://ymroueh.me/), 
[Kristjan Greenewald](https://kgreenewald.github.io/),
[Antonio Torralba](http://web.mit.edu/torralba/www/), and
[Stefanie Jegelka](https://people.csail.mit.edu/stefje/)
<br/>


## Prerequisites
- Python 3.7 
- NumPy
- Tensorflow 1.15.0
- [POT](https://pythonot.github.io/)

## PGDL Dataset
Download the dataset from the official Predicting Generalization in Deep Learning (PGDL) [repo](https://github.com/google-research/google-research/tree/master/pgdl). The evaluation tool and instructions can be found [here](https://competitions.codalab.org/competitions/25301#learn_the_details-get_starting_kit). The folders, ```kV-Margin``` and ```kV-GN-Margin```, contain the implementation of our k-variance normalized margins (see ```complexity.py```).

To evaluate our ```kV-Margin``` or ```kv-GN-Margin```,  download data and run
``` 
python ingestion_program/ingestion.py **path/to/inptu_data** **path/to/prediction** ingestion_program kV-Margin
```

To compute the score of your submission locally, you can run the scoring program:
``` 
python scoring_program/score.py **path/to/reference_data** **path/to/prediction** **path/to/output**
```

One can modify the ```shallow``` argument in ```complexity.py``` to choose shallow (1st) or deep layer (8th) feature to compute the margins.

## Results 
| random seed = 0 | task1 | task2 | task4 | task5 | task6 | task7 | task8 | task9 |
|----------|:---:|:----:|:---:|:---:|:---:|:---:|:---:|:---:|
|  kv-Margin 1st | 5.83 | 29.11 | 36.45 | 17.51 | 6.27 | 2.11 | 1.89 | 13.89 |
|  kv-Margin 8th | 29.72 | 27.19 | 6.01 | 15.10 | 0.81 | 1.71 | 0.37 | 1.43|
|  kv-GN-Margin 1st | 18.17  | 45.24 | 30.78 | 15.66 | 4.48 | 3.91 | 0.58 | 21.85 |
|  kv-GN-Margin 8th | 41.09  | 46.28 | 6.71 | 15.99 | 1.11 | 5.93 | 0.66 | 8.14 |

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{chuang2021measuring,
  title={Measuring Generalization with Optimal Transport},
  author={Chuang, Ching-Yao and Mroueh, Youssef and Greenewald, Kristjan and Torralba, Antonio and Jegelka, Stefanie},
  journal={arXiv preprint arXiv:2106.03314},
  year={2021}
}
```
For any questions, please contact Ching-Yao Chuang (cychuang@mit.edu).



