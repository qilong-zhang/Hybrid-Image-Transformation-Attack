# Hybrid Image Transformation Attack

This is Pytorch code for our paper [Practical No-box Adversarial Attacks with Training-free
Hybrid Image Transformation](https://arxiv.org/abs/2203.04607). In this paper, we move a step forward and show the existence of a **training-free** adversarial perturbation under the no-box threat model, which can be successfully used to attack different DNNs in real-time. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of the proposed no-box method. It attacks ten well-known models with a success rate of **98.13%** on average, which outperforms state-of-the-art no-box attacks by **29.39%**. Furthermore, our method is even competitive to mainstream transfer-based black-box attacks.



## Requirement

- python3.7
- torch 1.5.0
- torchvision 0.6.0
- numpy 1.16.6
- opencv-python 4.5.1.48
- timm 0.4.12
- matplotlib 3.3.4

## Implemenation
- Put the ImageNet dataset into "input/" whose structure like the following: 
  ```
  No-box-HIT-Attack
  |───input
  |   |───n01440764
  |   |───n01443537
  |   |───n01484850
  |   |───n01491361
  |   |───n01494475
  .....
  |   |───n15075141
  ```
  where each folder represents a category and `torchvision.datasets.ImageFolder` can automatically get their corresponding labels. For other dataset, you may need to write a dataloader by yourself.

- Run the below code to perform our HIT attack:

  ```python
  python HIT.py
  ```

## Citing this work
If you think our work is intersting or useful in your research, please consider citing:

```
@article{zhang2022hit,
  author    = {Qilong Zhang and
               Chaoning Zhang and
               Chaoqun Li and
               Jingkuan Song and
               Lianli Gao and
               Heng Tao Shen},
  title     = {Practical No-box Adversarial Attacks with Training-free Hybrid Image
               Transformation},
  journal   = {CoRR},
  volume    = {abs/2203.04607},
  year      = {2022}
}
```




