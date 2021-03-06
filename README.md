<h1 align="center">
  <b>VAEs</b><br> 
  <b>Jax | Flux | PyTorch</b><br> 
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.10-2BAF2B.svg" /></a>
       <a href= "https://fluxml.ai/">
        <img src="https://img.shields.io/badge/Flux-v0.12.8-red" /></a>
       <a href= "https://github.com/google/jax">
        <img src="https://img.shields.io/badge/Jax-v0.1.75-yellow" /></a>
       <a href= "https://github.com/BeeGass/VAEs/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
         <a href= "http://twitter.com/intent/tweet?text=Readable-VAEs:%20A%20Collection%20Of%20VAEs%20Written%20In%20PyTorch%20And%20Jax%3A&url=https://github.com/BeeGass/Readable-VAEs">
        <img src="https://img.shields.io/twitter/url/https/shields.io.svg?style=social" /></a>

</p>

A collection of Variational AutoEncoders (VAEs) I have implemented in [jax](https://github.com/google/jax)/[flax](https://github.com/google/flax), [flux](https://fluxml.ai/) and [pytorch](https://PyTorch.org/) with particular effort put into readability and reproducibility. 

## Python 
### Requirements For Jax
- Python >= 3.8
- jax

#### Installation
```
$ git clone https://github.com/BeeGass/Readable-VAEs.git
```

#### Usage
```
$ cd Readable-VAEs/vae-jax
$ python main.py 
```

### Requirements For PyTorch
- PyTorch >= 1.10

#### Usage
```
$ cd Readable-VAEs/vae-pytorch
$ python main.py 
```

## Julia
### Requirements For Flux
- TODO
- TODO

#### Usage
```
$ cd Readable-VAEs/vae-flux
$ # TBA 
```
--- 

**Config File Template**
```yaml
TBA
```

**Weights And Biases Integration**
```
TBA
```

----
<h2 align="center">
  <b>Results</b><br>
</h2>


| Model           | PyTorch  | Jax/Flax  |  Flux   | Config  | Paper                                              | Reconstruction | Samples |
|:--------------- |:--------:|:---------:|:-------:|:-------:|:-------------------------------------------------- |:--------------:|:-------:|
| VAE             | &#9745;  |  &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1312.6114)            |    **TBA**     | **TBA** |
| Beta-VAE        | &#9744;  |  &#9744;  | &#9744; | &#9744; | [Link](https://openreview.net/forum?id=Sy2fzU9gl)  |    **TBA**     | **TBA** |
| Conditional VAE | &#9744;  |  &#9744;  | &#9744; | &#9744; | [Link](https://openreview.net/forum?id=rJWXGDWd-H) |    **TBA**     | **TBA** |
| VQ-VAE-2        | &#9744;  |  &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1906.00446)           |    **TBA**     | **TBA** |


### Citation
```bib
@software{Gass_Readable-VAEs_2021,
  author = {Gass, B.A., Gass, B.A.},
  doi = {10.5281/zenodo.1234},
  month = {12},
  title = {{Readable-VAEs}},
  url = {https://github.com/BeeGass/Readable-VAEs},
  version = {1.0.0},
  year = {2021}
}
```
