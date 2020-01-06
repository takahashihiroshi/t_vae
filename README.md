# Student-t Variational Autoencoder for Robust Density Estimation  
This is a pytorch implementation of the following paper [[URL]](https://www.ijcai.org/proceedings/2018/374):  
```
@inproceedings{takahashi2018student,
  title={Student-t Variational Autoencoder for Robust Density Estimation.},
  author={Takahashi, Hiroshi and Iwata, Tomoharu and Yamanaka, Yuki and Yamada, Masanori and Yagi, Satoshi},
  booktitle={IJCAI},
  pages={2696--2702},
  year={2018}
}
```
Please read license.txt before reading or using the files.  

## Prerequisites  
Please install `python>=3.6`, `torch`, `numpy` and `scikit_learn`.  

## Usage  
```
usage: main.py [-h] [--dataset DATASET] [--decoder DECODER]
               [--learning_rate LEARNING_RATE] [--seed SEED]
```
- You can choose the `dataset` from following datasets: `SMTP`.  
  - We are preparing other datasets.  
- You can choose the `decoder` of the VAE from `normal` or `student-t`.  
- You can also change the random `seed` of the training and `learning_rate` of the optimizer (Adam).  


## Example  
SMTP with Gaussian decoder:  
```
python main.py --dataset SMTP --decoder normal
```
SMTP with Student-t decoder:  
```
python main.py --dataset SMTP --decoder student-t
```

## Output  
- After the training, the mean of log-likelihood for test dataset will be displayed.  
- The detailed information of the training and test will be saved in `npy` directory.  
