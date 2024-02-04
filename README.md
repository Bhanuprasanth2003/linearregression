Report on Linear Regression Model Implementation

Cost Function Value and Learning Parameters Values after Convergence

After implementing the linear regression model using batch gradient descent with a learning rate of 0.5, the cost function value after convergence was found to be approximately 0.000342, and the learning parameters (θ0 and θ1) were [0.99629669, -0.00196178]. The convergence criteria used was the change in the cost function being less than 1e-5 between consecutive iterations.


Advantage of Averaging the Cost

The cost function used in this assignment is the mean squared error, and averaging this cost over the entire dataset has the advantage of normalizing the cost. Averaging helps make the cost function independent of the number of training examples, making it easier to compare models trained on different datasets or with different sizes. It provides a standardized measure of the model's performance.


Cost Function vs Iteration Graph

A plot of the cost function against iteration was created to visualize the convergence of the model. As shown in the graph, the cost decreases rapidly in the initial iterations and then gradually stabilizes, indicating convergence. The graph helps in understanding the model's learning process and the effectiveness of the chosen learning rate.



Dataset and Linear Regression Fit Visualization

The dataset and the linear regression fit were visualized in a scatter plot. The red line represents the linear regression fit obtained from the model. The fit shows a clear trend capturing the relationship between the independent and dependent variables.



Testing with Different Learning Rates

The linear regression model was tested with three different learning rates (0.005, 0.5, 5). The cost function vs iteration graphs for each learning rate were plotted. It was observed that a learning rate of 0.5 provided a smooth and effective convergence, while lower and higher learning rates led to slower convergence and oscillations, respectively.



Gradient Descent Methods Comparison

Batch gradient descent, stochastic gradient descent, and mini-batch gradient descent with a batch size of 10 were compared. The cost function vs iteration graphs showed that batch gradient descent provides a smooth and steady convergence, while stochastic gradient descent and mini-batch gradient descent exhibit more fluctuations. The choice of batch size in mini-batch gradient descent can affect the convergence behavior.


In summary, the implemented linear regression model demonstrates effective convergence with a learning rate of 0.5, and the choice of learning rate significantly influences the model's performance. The visualization of the cost function, dataset, and different gradient descent methods aids in understanding the model's behavior and guiding further optimizations.
