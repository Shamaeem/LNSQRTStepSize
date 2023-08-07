### Code & Usage
`src` folder contains codes for training a deep neural network to do image classification on FashionMNIST and CIFAR10. You can train models with the `main.py` script, with hyper-parameters being specified as flags.

After obtaining the results, to see the comparison, use `draw_comps.py` by specifying the logs folder, for example:
```
python draw_comps.py --logs-folder ./logs/CIFAR10 --fig-type others
```



#### FashionMNIST
```
python ./src/main.py --optim-method SGD --eta0 0.007 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 100 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SGD_1sqrt_Decay --eta0 0.05 --alpha 0.00653 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method Adam --eta0 0.0009 --weight-decay 0.0001 --train-epochs 100 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SGD_Cosine_Decay --eta0 0.05 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SLS-Armijo1 --eta0 0.5 --c 0.1 --train-epochs 100 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

python ./src/main.py --optim-method SGD_1sqrtlnt_Decay --eta0 0.05 --\alpha  0.0253 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNISTStep  --log_step_length_folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data

```


#### CIFAR10
```
python ./src/main.py --optim-method SGD --eta0 0.07 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SGD_1sqrt_Decay --eta0 0.2 --alpha 0.079079 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method Adam --eta0 0.0009 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SGD_Cosine_Decay --eta0 0.25 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SGD_1sqrtlnt_Decay --eta0 0.15  --\alpha 0.25 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data

python ./src/main.py --optim-method SLS-Armijo0 --eta0 2.5 --c 0.1 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR10 --dataset CIFAR10 --dataroot ./data
```


#### mushrooms
```
python ./src/trainval.py - --exp_group_list mushrooms --datadir ../data  --optim-method SGD_1sqrt_Decay --eta0 0.05 --alpha 0.00653 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/mushrooms --dataset mushrooms --dataroot ./data `

python ./src/trainval.py  --exp_group_list mushrooms --datadir ../data  --optim-method SGD_Stage_Decay --eta0 0.04 --alpha 0.1 --milestones 12000 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/mushrooms --dataset mushrooms --dataroot ./data 

python ./src/trainval.py  --exp_group_list mushrooms --datadir ../data  --optim-method SGD_Stage_Decay --eta0 0.04 --alpha 0.1 --milestones 9000 15000 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/mushrooms --dataset mushrooms --dataroot ./data 


python ./src/trainval.py  --exp_group_list mushrooms --datadir ../data  --optim-method SGD --eta0 0.007 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/mushrooms --dataset mushrooms --dataroot ./data

python ./src/trainval.py --exp_group_list mushrooms --datadir ../data --optim-method SGD_1sqrtlnt_Decay --eta0 0.05 --alpha 0.00001 --nesterov --momentum 0.9 --weight-decay 0.00001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/mushrooms --dataset mushrooms --dataroot ./data

python ./src/trainval.py --exp_group_list mushrooms --datadir ../data --optim-method SGD_ReduceLROnPlateau --eta0 0.04 --alpha 0.5 --patience 3 --threshold 0.001 --nesterov --momentum 0.9 --weight-decay 0.001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/mushrooms --dataset mushrooms --dataroot ./data


python ./src/trainval.py --exp_group_list mushrooms --datadir ../data --optim-method SGD_1t_Decay --eta0 0.05 --alpha 0.000384 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/mushrooms --dataset mushrooms --dataroot ./data

python ./src/trainval.py --exp_group_list mushrooms --datadir ../data --optim-method SGD_Cosine_Decay --eta0 0.05 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/mushrooms --dataset mushrooms --dataroot ./data
```

#### a1a
```
python ./src/trainval.py - --exp_group_list a1a --datadir ../data  --optim-method SGD_1sqrt_Decay --eta0 0.05 --alpha 0.00653 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a1a --dataset a1a --dataroot ./data `

python ./src/trainval.py  --exp_group_list a1a --datadir ../data  --optim-method SGD_Stage_Decay --eta0 0.04 --alpha 0.1 --milestones 12000 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a1a --dataset a1a --dataroot ./data 

python ./src/trainval.py  --exp_group_list a1a --datadir ../data  --optim-method SGD_Stage_Decay --eta0 0.04 --alpha 0.1 --milestones 9000 15000 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a1a --dataset a1a --dataroot ./data 


python ./src/trainval.py  --exp_group_list a1a --datadir ../data  --optim-method SGD --eta0 0.007 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a1a --dataset a1a --dataroot ./data

python ./src/trainval.py --exp_group_list a1a --datadir ../data --optim-method SGD_1sqrtlnt_Decay --eta0 0.05 --alpha 0.00001 --nesterov --momentum 0.9 --weight-decay 0.00001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a1a --dataset a1a --dataroot ./data

python ./src/trainval.py --exp_group_list a1a --datadir ../data --optim-method SGD_ReduceLROnPlateau --eta0 0.04 --alpha 0.5 --patience 3 --threshold 0.001 --nesterov --momentum 0.9 --weight-decay 0.001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a1a --dataset a1a --dataroot ./data


python ./src/trainval.py --exp_group_list a1a --datadir ../data --optim-method SGD_1t_Decay --eta0 0.05 --alpha 0.000384 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a1a --dataset a1a --dataroot ./data

python ./src/trainval.py --exp_group_list a1a --datadir ../data --optim-method SGD_Cosine_Decay --eta0 0.05 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a1a --dataset a1a --dataroot ./data
```
#### a2a
```
python ./src/trainval.py - --exp_group_list a2a --datadir ../data  --optim-method SGD_1sqrt_Decay --eta0 0.05 --alpha 0.00653 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a2a --dataset a2a --dataroot ./data `

python ./src/trainval.py  --exp_group_list a2a --datadir ../data  --optim-method SGD_Stage_Decay --eta0 0.04 --alpha 0.1 --milestones 12000 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a2a --dataset a2a --dataroot ./data 

python ./src/trainval.py  --exp_group_list a2a --datadir ../data  --optim-method SGD_Stage_Decay --eta0 0.04 --alpha 0.1 --milestones 9000 15000 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a2a --dataset a2a --dataroot ./data 


python ./src/trainval.py  --exp_group_list a2a --datadir ../data  --optim-method SGD --eta0 0.007 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a2a --dataset a2a --dataroot ./data

python ./src/trainval.py --exp_group_list a2a --datadir ../data --optim-method SGD_1sqrtlnt_Decay --eta0 0.05 --alpha 0.00001 --nesterov --momentum 0.9 --weight-decay 0.00001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a2a --dataset a2a --dataroot ./data

python ./src/trainval.py --exp_group_list a2a --datadir ../data --optim-method SGD_ReduceLROnPlateau --eta0 0.04 --alpha 0.5 --patience 3 --threshold 0.001 --nesterov --momentum 0.9 --weight-decay 0.001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a2a --dataset a2a --dataroot ./data


python ./src/trainval.py --exp_group_list a2a --datadir ../data --optim-method SGD_1t_Decay --eta0 0.05 --alpha 0.000384 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a2a --dataset a2a --dataroot ./data

python ./src/trainval.py --exp_group_list a2a --datadir ../data --optim-method SGD_Cosine_Decay --eta0 0.05 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 1000 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/a2a --dataset a2a --dataroot ./data
```
```
