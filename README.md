# VAE_101
Here I am implementing various types of VAE algorithm. I'll add more!


![Result](https://github.com/tiassap/VAE_101/blob/main/output/vanilla_VAE-01/output.png?raw=true)
_Digit output on MNIST training dataset_

To train:
```
python run.py --config=<config_file> --train
```

Example: `python run.py --config=vanilla_VAE --train`


Modify parameter in `config/*.yml` file, for example `vanilla_VAE.yml`:
```
config_no: &no "01"
name: &name "vanilla_VAE"

hyperparameters:
  learning_rate: 0.0001
  batch_size: 32
  num_epoch: 50
  in_feature_dim: 784
  hidden_dim: 512
  latent_dim: 2
```