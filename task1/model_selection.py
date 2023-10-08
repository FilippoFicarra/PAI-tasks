import torch
from gpytorch.kernels import ScaleKernel, MaternKernel, LinearKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np

from solution import extract_city_area_information, cost_function, cluster_data
from solution_torch import ExactGPModel

# Variables
ADJ_FACTOR = 2.
TRAINING_ITERATIONS = 500


def train(x_train, y_train, model, likelihood):
    # Set model and likelihood for training
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.25)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for i in range(TRAINING_ITERATIONS):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        if (i + 1) % 50 == 0:
            print(
                f"Iter {i + 1}/{TRAINING_ITERATIONS} - Loss: {loss.item()} noise: {model.likelihood.noise.item()}")
        optimizer.step()


def predict_and_evaluate(x_test, x_test_AREA, ground_truth, model, likelihood):
    # Set model and likelihood for evaluation
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        y_preds = likelihood(model(x_test))
        gp_mean = y_preds.mean
        gp_std = torch.sqrt(y_preds.variance).numpy()

    predictions = gp_mean.numpy()

    # Adjust predictions
    for i in range(x_test_AREA.shape[0]):
        if x_test_AREA[i]:
            # Adjust prediction by shifting it by a value proportional to the standard deviation
            predictions[i] += ADJ_FACTOR * gp_std[i]

    return cost_function(ground_truth, predictions, x_test_AREA)


def k_fold_cross_validation(x_train, y_train, x_train_AREA, k):
    # Define indeces of each fold
    N = x_train.shape[0]
    N_samples_fold = int(N / k)
    indeces = np.array(list(range(N)))
    folds = []

    for i in range(k):
        # Sample indeces of split
        folds.append(np.array(np.random.choice(indeces, N_samples_fold, replace=False)))
        # Remove sampled indeces from list of available indeces
        indeces = np.setdiff1d(indeces, folds[len(folds) - 1])

    # Define kernels to test
    kernels = {"c*rbf*rbf": [ScaleKernel(RBFKernel() * RBFKernel()) for _ in range(k)],
               "c*(rbf+rbf)": [ScaleKernel(RBFKernel() + RBFKernel()) for _ in range(k)],
               "l+c*rbf": [LinearKernel() + ScaleKernel(RBFKernel()) for _ in range(k)],
               "c*matern*matern": [ScaleKernel(MaternKernel()*MaternKernel()) for _ in range(k)],
               "c*(matern+matern)": [ScaleKernel(MaternKernel() + MaternKernel()) for _ in range(k)],
               "l+c*matern": [LinearKernel() + ScaleKernel(MaternKernel()) for _ in range(k)]}

    mean_costs = {}
    all_costs = {}
    for name, kernel in kernels.items():
        # Perform cross validation using the current kernel
        print(f"Evaluating kernel {name}...")
        kernel_costs = np.zeros(k)
        for j in range(k):
            print(f"Evaluation {j+1}/{k} for {name}.")

            # Define current training and validation sets. folds[i] is used for testing
            k_fold_test_x, k_fold_test_y = x_train[folds[j], :], y_train[folds[j]]
            k_fold_train_x, k_fold_train_y = np.delete(train_x, folds[j], 0), np.delete(train_y, folds[j])
            k_fold_test_x_AREA = x_train_AREA[folds[j]]

            # Remove mean from y and train model.
            # mean_y = k_fold_train_y.mean()
            # k_fold_train_y = k_fold_train_y - mean_y

            # Transform arrays to tensors
            k_fold_test_x = torch.tensor(k_fold_test_x, dtype=torch.float32)
            k_fold_train_x = torch.tensor(k_fold_train_x, dtype=torch.float32)
            k_fold_train_y = torch.tensor(k_fold_train_y, dtype=torch.float32)

            # Define model and change kernel
            likelihood = GaussianLikelihood()
            model = ExactGPModel(k_fold_train_x, k_fold_train_y, likelihood)
            model.change_kernel(kernel[j])

            # Train model
            train(k_fold_train_x, k_fold_train_y, model, likelihood)

            # Predict and evaluate using cost function
            cost = predict_and_evaluate(k_fold_test_x, k_fold_test_x_AREA, k_fold_test_y, model, likelihood)
            print(f"Result evaluation {j+1}/{k} for {name}: {cost}.")
            kernel_costs[j] = cost

        # Add mean to list of costs
        print(f"Mean cost for {name} after {k}-fold: {kernel_costs.mean()}.")
        mean_costs[name] = kernel_costs.mean()
        all_costs[name] = kernel_costs

    # Print all costs and get name of best kernel
    print(all_costs)
    print(f"The best is {min(mean_costs, key=mean_costs.get)}.")


if __name__ == "__main__":
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)

    # Subsample train set using k means
    indeces, train_x, train_y = cluster_data(train_y, train_x_2D, k=4500, plot=False)
    # Fit the model
    print('Cross validation on kernels...')
    k_fold_cross_validation(train_x, train_y, train_x_AREA[indeces], 3)
