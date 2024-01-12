# CVML


## Workflow

First define a physical model with a text file. For example, phi-4 theory can be saved as
scalar.Model(
geom=(4,),
nbeta=4,
nt=0,
m2=0.01,
lamda=0.01
)

Then make configurations for training using sample_scalar.cpp and convert it to the jax array. The form of the jax array is the array of each configurations, and the dimension should be (n_config, degree of freedom).

Finally, use cv.py to train the subtraction function to optimize variance of a particular observable.

Here is an example:
```
mkdir -p data
vi model.dat # and copy and paste the above model
make sample_scalar
./sample_scalar 4 4 0.01 0.01 100 2000 > data/sample.dat \& 
./converter.py '(4,4)'  data/sample.dat data/config.pickle
./cv.py data/model.dat data/cv.pickle data/config.pickle -i -l 1 -w 8 -lr 1e-3 -s -C 1000 # Terminal with CTRL-C
```