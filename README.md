# Visualizing Back Propagation

Graphical Representation of a neural network's training using stochastic gradient descent with one hidden layer where loss is squared distance. Includes ability to toggle sigmoid activation and relu activation. Used impworld from javalib (Northeastern CS2510) for graphics.

Input:
- The two inputs refer to the x and y point on a cartesian plane
- The output reflects whether or not point lies in region above y = x^2

Training Model:
- Used two hidden nodes to reflect the "two lines" that can be visualized as the defining traits of y=x^2
- Stochastic, randomly pulls points from training data to test, updating only when incorrectly classified
- Chain rule in gradient descent to update all weights, values can be seen below network
- Color coded lines to show intensity of weight (more stimulated, the darker)

![](Images%20and%20GIFS/Demonstration.gif)

