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
