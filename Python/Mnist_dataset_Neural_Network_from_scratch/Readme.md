## Results

+ In the end the cost was reduced to .0595 and the program correctly identified 9821/10000
+ I didn't break off a holdout set like done in the neural network book, so our training data was on 60k observations 

## Overall algorithm 

The approach used for this neural network, is momentum based mini batch stochastic gradient descent.   Stochastic Gradient decent has the computational advantage of allowing us to perform multiple parameter updates per epoch.   However, SGD does lead to heavy fluctuation in the cost estimate. Therefore, mini-batch SD tends to both reduce this error and provide us computational speedup.  Tuning is where the importance comes in.    

Momentum, allows for a faster learning rate.  Essentially large learning rates can cause us to have far too large fluctuations in our cost function, we can overshoot our minimums.    Andrew Ng describes momentum as providing a more direct path towards the minimum.  It allows us to move rapidly towards the minimum without overshooting it in vertical direction.  The below image the blue line is typical MBGD, where as the red line is how Momentum based MBGD finds the minimum  


![Image](https://github.com/cuny-sps-msda-data622-2018fall/fall2018-data622-001-project-justinherman42/blob/master/from.PNG)



##  Cost function

  Cross entropy loss, is used as its the cost function used with the softmax activation layer.  Cross entropy loss is computed on probability output distributions, which is what softmax gives us as an output.  Cross entropy has other advantages of avoiding the learning slowdown from saturation involved with the quadratic cost function.  
Certain tricks were added into the algorithm to  help avoid learning slowdown and saturation.  The algorithm sets up initial weights that are multiplied by $1/ \sqrt{N}$ which helps with vanishing and exploding gradients.  If our gradients become too large or too small, it slows down learning.  Setting weights in this way helps to prevent this from happening.  The activation function used for our input layer, is the sigmoid function.  The output layer, is a softmax function.  The softmax function, converts our output layer to a vector of probabilities summing to 1.          

## Hyperparameters

+ For Batch optimization, Andrew Ng explained that computers optimize around 2^n, so  choosing batch sizes of 2^n can help speed up learning.  After experimenting with changing around the original script I stayed with the initial batch size of 128.
+ The hyper parameter of .9 is the most typically used value for the momentum parameter.
+ I chose 1 layer.  When I attempted to code in a second layer, I actually saw worse results, although I didn't attempt to optimize it
+ Neurons-185.  I read online several recommendations for tuning amount of neurons having to do with the relative size of input and output layers, but through trial and error this seemed to give the best results.
+ Learning rate seemed to optimize around 4.5
  + Learning rate is the most important parameter to tune, and typically gives the biggest benefit in minimizing cost
+ I added in a learning rate decay(.02) and attempted to tune that as well.  This was all done via trial and error and I am not sure this was necessary given the momentum parameter, but it appeared to give me better results  

+ Final parameters
  + neurons = 185
  + my_learning_rate = 4.5
  + beta = .9( momentum)
  + batch_size = 128
  + learning_rate_decay = .02

