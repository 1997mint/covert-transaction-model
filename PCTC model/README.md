## SeqGAN
通过SeqGAN生成比特币交易的金额和手续费。细节请阅读[paper](https://arxiv.org/abs/1609.05473)

## 生成
```
python main.py
```
生成后的语句保存在test.txt中，包含交易总金额，手续费以及转给其中一个输出地址的金额。一次可生成1000条数据。

## 部分生成语句
 t o t a l 5 0 8 8 0 8 4 , f e e s 4 5 2 0 , v a l u e 5 3 4 2
t o t a l 1 3 4 9 2 5 0 6 , f e e s 1 8 9 8 4 , v a l u e 3 5 0 7 1 9
t o t a l 1 8 6 7 1 9 , f e e s 1 8 3 1 , v a l u e 7 8 9 5 2
t o t a l 4 4 9 5 4 , f e e s 2 2 7 9 1 7 , v a l u e 4 2 1 0 2 2
t o t a l 2 2 7 1 3 8 4 3 3 3 , f e e s 4 9 7 2 , v a l u e 8 9 5 9 9 9
t o t a l 1 1 7 8 5 0 0 0 0 0 , f e 0 0 , v t a l 7 8 5 8 1 2 8 , v a l u e 9 6 1 8 8 0 4 2 6
t o t a l 6 3 9 6 8 9 9 1 3 , f e e s 3 5 0 0 0 , v a l u e 3 4 5 1 9 8
t o t a l 9 6 9 9 9 9 9 9 , f e e s 1 1 2 5 0 , v a l u e 6 6 6 3 1 3 9
t o t a l 8 1 2 4 8 3 4 , f e e s 3 2 4 7 , v a l u e 6 0 8 3 0 0
t o t a l 1 6 8 4 8 8 0 1 8 2 , f e e s 3 6 8 0 , v a l u e 5 3 4 9 7 1 2              


## Requirements
 - Python>3.5
 - Keras==2.2.4
 - tensorflow==1.12.0

## generate transactions的数据为使用GAN生成的数据后产生的模拟交易
