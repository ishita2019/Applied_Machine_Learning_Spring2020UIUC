# Applied_Machine_Learning_Spring2020UIUC
This repo contains all my machine learning work in Python, R, Neural Networks and Deep Neural Networks

Project 6:
Log -Log linear regression model to predict concentration of sulfate in the blood of a Baboon as a function of Time:

A log-log linear regression model was built to predict the concentration of Sulfate in the Blood of a Baboon as a function of time. The data is original coordinates were transformed in log-scale for the linear regression, and the model regression curve shows better fit to the data points than a simple linear regression model against the same data.

Our purpose was to verify the residual vs Fitted plot for both the log-log linear regression and simple linear regression against the data point in both original coordinates and in log coordinates.

The result shows the better residual vs fitted plot in log-log linear regression in the log coordinates.
The simple linear regression model built to predict body mass as a function of various body parts diameter performs poorly than a regression model where regressing the cube root of the body mass as a function of these diameters.

The plot of the residual against the fitted values has proved it after performing the project.


Project_8 :
Adversarial machine learning problems:

Projects

Adversarial machine learning problems
Apr 2020 – May 2020

Project descriptionThe project takes the input images as images of the digits from the MNIST data set and randomly sample 10 images from the data set. To each image a white noise is introduced to set the image as an adversarial image such that a neural network model misclassifies the image to our goal label instead of original label.
We first generate a random image x using the normal distribution,
x = np.random.normal(.5, .3, (784, 1))
We then use the following update equation to iteratively update the value of
x at each step:
x -= eta * (d + lam * (x - x_target))

Where eta is the step size, lam is the regularization parameter (weight of
the target image x_target) and d is the gradient of the loss w.r.t. x. We run it
for 1000 iterations or steps.
In our experiments we set eta=0.1, lam=0.05, steps=1000

Result:
In each case the image
was misclassified as the next image as in the next number (e.g. 1 was classified as 2).

