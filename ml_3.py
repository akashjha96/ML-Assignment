import numpy as np
import matplotlib.pyplot as plt

independent_var = np.array([9.1, 8, 9.1, 8.4, 6.9, 7.7, 15.6, 7.3, 7, 7.2, 10.1, 11.5, 7.1, 10, 8.9, 7.9,
                            5.6, 6.3, 6.7, 10.4, 8.5, 7.4, 6.3, 5.4, 8.9, 9.4, 7.5, 11.9, 7.8, 7.4, 10.8,
                            10.2, 6.2, 7.7, 13.7, 8, 6.7, 6.7, 7, 8.3, 7.4, 9.9, 6.1, 7, 5.4, 10.7, 7.6,
                            8.9, 9.2, 6.6, 7.2, 8, 7.8, 7.9, 7, 7, 7.6, 9.1, 9, 7.9, 6.6, 11.9, 6.5, 7.1,
                            8.8, 7.5, 7.7, 6, 10.6, 6.6, 8.2, 7.9, 7.1, 5.6, 6.4, 7.5, 9.8, 7, 10.5, 7.1,
                            6.2, 6.5, 7.7, 7.2, 9.3, 8.5, 7.7, 6.8, 7.8, 8.7, 9.6, 7.2, 9.3, 8.1, 6.6, 7.8,
                            10.2, 6.1, 7.3, 7.3])
dependent_var = np.array([0.99523, 0.99007, 0.99769, 0.99386, 0.99508, 0.9963, 1.0032, 0.99768, 0.99584,
                          0.99609, 0.99774, 1.0003, 0.99694, 0.99965, 0.99549, 0.99364, 0.99378, 0.99379,
                          0.99524, 0.9988, 0.99733, 0.9966, 0.9955, 0.99471, 0.99354, 0.99786, 0.9965,
                          0.9988, 0.9964, 0.99713, 0.9985, 0.99565, 0.99578, 0.9976, 1.0014, 0.99685,
                          0.99648, 0.99472, 0.99914, 0.99408, 0.9974, 1.0002, 0.99402, 0.9966, 0.99402,
                          1.0029, 0.99718, 0.9986, 0.9952, 0.9952, 0.9972, 0.9976, 0.9968, 0.9978, 0.9951,
                          0.99629, 0.99656, 0.999, 0.99836, 0.99396, 0.99387, 1.0004, 0.9972, 0.9972,
                          0.99546, 0.9978, 0.99596, 0.99572, 0.9992, 0.99544, 0.99747, 0.99668, 0.9962,
                          0.99346, 0.99514, 0.99476, 1.001, 0.9961, 0.99598, 0.99608, 0.9966, 0.99732,
                          0.9962, 0.99546, 0.99738, 0.99456, 0.9966, 0.99553, 0.9984, 0.9952, 0.997,
                          0.99586, 0.9984, 0.99542, 0.99655, 0.9962, 0.9976, 0.99464, 0.9983, 0.9967])

# Choose a suitable learning rate
lr = 0.3
epochs = 1000  # You can adjust this based on convergence

# Normalize the dataset
independent_var = (independent_var - np.mean(independent_var)) / np.std(independent_var)
dependent_var = (dependent_var - np.mean(dependent_var)) / np.std(dependent_var)

# Function to perform gradient descent (batch, stochastic, or mini-batch)


def gradient_descent(independent_var, dependent_var, theta0, theta1, lr, epochs, batch_size=None):
    cost_history = []
    m = len(independent_var)

    for epoch in range(epochs):
        # Shuffle the data for stochastic and mini-batch GD
        if batch_size is not None:
            indices = np.random.permutation(m)
            independent_var = independent_var[indices]
            dependent_var = dependent_var[indices]

        for i in range(0, m, batch_size) if batch_size is not None else range(m):
            end_index = i + batch_size if batch_size is not None else i + 1
            x_batch = independent_var[i:end_index]
            y_batch = dependent_var[i:end_index]

            predicted_values = theta0 + theta1 * x_batch
            cost = np.mean((predicted_values - y_batch) ** 2)
            gradient_theta0 = np.mean(predicted_values - y_batch)
            gradient_theta1 = np.mean((predicted_values - y_batch) * x_batch)

            theta0 = theta0 - lr * gradient_theta0
            theta1 = theta1 - lr * gradient_theta1

        cost_history.append(cost)

    return cost_history, theta0, theta1

# Batch Gradient Descent
theta0_batch = 0
theta1_batch = 0
cost_history_batch, _, _ = gradient_descent(
    independent_var, dependent_var, theta0_batch, theta1_batch, lr, epochs)

# Stochastic Gradient Descent
theta0_sgd = 0
theta1_sgd = 0
cost_history_sgd, _, _ = gradient_descent(
    independent_var, dependent_var, theta0_sgd, theta1_sgd, lr, epochs, batch_size=1)

# Mini-Batch Gradient Descent
theta0_mini_batch = 0
theta1_mini_batch = 0
batch_size_mini_batch = 10  # Adjust batch size as needed
cost_history_mini_batch, _, _ = gradient_descent(
    independent_var, dependent_var, theta0_mini_batch, theta1_mini_batch, lr, epochs, batch_size=batch_size_mini_batch)

# Plot cost function vs. iteration for different gradient descent methods
plt.plot(range(len(cost_history_batch)),
         cost_history_batch, label='Batch Gradient Descent')
plt.plot(range(len(cost_history_sgd)), cost_history_sgd,
         label='Stochastic Gradient Descent')
plt.plot(range(len(cost_history_mini_batch)), cost_history_mini_batch,
         label=f'Mini-Batch Gradient Descent (Batch Size = {batch_size_mini_batch})')

# Plot settings
plt.xlabel('Iteration')
plt.ylabel('Cost Function')
plt.title('Cost Function vs. Iteration for Different Gradient Descent Methods')
plt.legend()
plt.show()

# Print cost for debugging
print("Batch GD Final Cost:", cost_history_batch[-1])
print("Stochastic GD Final Cost:", cost_history_sgd[-1])
print(
    f"Mini-Batch GD (Batch Size = {batch_size_mini_batch}) Final Cost:", cost_history_mini_batch[-1])