# Measuring Generalization with Optimal Transport

<p align='left'>
<img src='https://github.com/chingyaoc/kV-Margin/blob/main/fig.png?raw=true' width='700'/>
</p>

Understanding the generalization of deep neural networks is one of the most important tasks in deep learning. Although much progress has been made, theoretical error bounds still often behave disparately from empirical observations. In this work, we develop margin-based generalization bounds, where the margins are normalized with optimal transport costs between independent random subsets sampled from the training distribution. In particular, the optimal transport cost can be interpreted as a generalization of variance which captures the structural properties of the learned feature space. Our bounds robustly predict the generalization error, given training data and network parameters, on large scale datasets. Theoretically, we demonstrate that the concentration and separation of features play crucial roles in generalization, supporting empirical results in the literature.


**Measuring Generalization with Optimal Transport** arXiv [[paper]](https://arxiv.org/abs/2007.00224)
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
Download the dataset from the official Predicting Generalization in Deep Learning (PGDL) [repo](https://github.com/google-research/google-research/tree/master/pgdl). The evaluation tool and instructions can be found [here](https://competitions.codalab.org/competitions/25301#learn_the_details-get_starting_kit).

To evaluate our ```kV-Margin``` or ```kv-GN-Margin```,  download data and run
``` 
python ingestion_program/ingestion.py **path/to/inptu_data** **path/to/prediction** ingestion_program kV-Margin
```

To compute the score of your submission locally, you can run the scoring program:
``` 
python scoring_program/score.py **path/to/reference_data** **path/to/prediction** **path/to/output**
```

### Results 
| random seed = 0 | task1 | task2 | task4 | task5 | task6 | task7 | task8 |
|----------|:---:|:----:|:---:|:---:|:---:|:---:|:---:|
|  kv-Margin 1st | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
|  kv-Margin 8th | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
|  kv-GN-Margin 1st |0.00  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
|  kv-GN-Margin 8th |0.00  | 0.00 | 0.00 | 0.00 | 0.0 | 0.00 | 0.00 |

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{chuang2021measuring,
  title={Measuring Generalization with Optimal Transport},
  author={Chuang, Ching-Yao and Robinson, Joshua and Lin, Yen-Chen and Torralba, Antonio and Jegelka, Stefanie},
  journal={arXiv},
  year={2021}
}
```
For any questions, please contact Ching-Yao Chuang (cychuang@mit.edu).



