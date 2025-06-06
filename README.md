# Neural control variates for lattice field theory

## Papers

If you use this code or a derivative of it, please consider citing one or more of the following papers.

- [Leveraging neural control variates for enhanced precision in lattice field theory](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.094519)
- [Control variates with neural networks](https://arxiv.org/abs/2501.14614)
- [Training neural control variates using correlated configurations](https://arxiv.org/abs/2505.07719)


## Workflow

First define a model with a text file. For example, phi-4 theory can be saved as
```
scalar.Model(
geom=(4,4),
m2=0.01,
lamda=0.01
)
```

Then make configurations for training using Monte Carlo in `mc`. The dimension of the configuration should be (n_config, degree of freedom).

Finally, use one of `cv.py` to train the subtraction function to optimize variance of a particular observable.

Here is an example:
```
mkdir -p data
vi model.dat # and copy and paste the above model
cd mc
make sample_scalar_2d
cd ../
./mc/sample_scalar 4 4 0.01 0.01 100 2000 data/sample.bin & 
./cv_scalar.py data/model.dat data/cv.pickle data/sample.bin -i -l 1 -w 8 -lr 1e-3 -s -C 1000 # Terminate with CTRL-C
```