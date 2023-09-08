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

Then make configurations for training using sample.py. Finally, use cv.py to train the subtraction function to optimize variance of a particular observable.


