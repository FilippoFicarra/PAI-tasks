import os
import time
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from matplotlib import cm
from sklearn.cluster import KMeans

# Variables.
# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300

# Cost function constants.
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

ADJ_FACTOR_AREA_1 = 1.55
TRAINING_ITERATIONS = 1000
NUM_SAMPLES = 9000


# Functions.
def plot_data(x: np.ndarray, y: np.ndarray):
    """
    Generate scatter plot of data.
    @param x: 2D coordinates of point.
    @param y: y values.
    """

    # Plot.
    plt.figure(figsize=(20, 20))
    range_y = (np.min(y), np.max(y))
    y_0_1 = (y - np.min(y)) / (np.max(y) - np.min(y))
    cmap = cm.get_cmap('coolwarm')
    color = cmap(y_0_1)[:, :3]

    plt.scatter(x[:, 0], x[:, 1], c=color)
    plt.suptitle(f"Range of y: ({range_y[0]}, {range_y[1]})")
    plt.show()


def cluster_data(train_x_2D: np.ndarray, train_y: np.ndarray, k: int = 5000, plot=True):
    """
    Clusterize data to reduce training size.
    @param train_x_2D: training data.
    @param train_y: training labels.
    @param k: number of samples returned.
    @param plot: bool to control plotting of samples distribution.
    @return: new indeces, new training set and set of training labels, in this order.
    """

    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    kmeans.fit(train_x_2D, train_y)

    def find_nearest_points_to_centroids(centroids, data_points):
        indices = []

        for centroid in centroids:
            distances = np.linalg.norm(data_points - centroid, axis=1)

            index = np.argmin(distances)

            indices.append(index)

        return indices

    x_train_new_indices = sorted(find_nearest_points_to_centroids(kmeans.cluster_centers_, train_x_2D))

    x_train_new = train_x_2D[x_train_new_indices]
    y_train_new = train_y[x_train_new_indices]

    if plot:
        plot_data(x_train_new, y_train_new)

    return x_train_new_indices, x_train_new, y_train_new


def cost_function(ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray) -> float:
    """
    Calculate the cost of a set of predictions.
    @param ground_truth: ground truth pollution levels as a 1d NumPy float array.
    @param predictions: predicted pollution levels as a 1d NumPy float array.
    @param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,).
    @return: total cost of all predictions as a single float.
    """

    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost.
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction.
    mask = (predictions < ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average.
    return np.mean(cost * weights)


def is_in_circle(coor, circle_coor):
    """
    Check if a coordinate is inside a circle.
    @param coor: 2D coordinate.
    @param circle_coor: 3D coordinate of the circle center and its radius.
    @return:  True if the coordinate is inside the circle, False otherwise.
    """

    return (coor[0] - circle_coor[0]) ** 2 + (coor[1] - circle_coor[1]) ** 2 < circle_coor[2] ** 2


def determine_city_area_idx(visualization_xs_2D):
    """
    Determine the city_area index for each coordinate in the visualization grid.
    @param visualization_xs_2D: 2D coordinates of the visualization grid.
    @return: 1D array of city_area indexes.
    """

    # Circles coordinates.
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                        [0.79915856, 0.46147936, 0.1567626],
                        [0.26455561, 0.77423369, 0.10298338],
                        [0.6976312, 0.06022547, 0.04015634],
                        [0.31542835, 0.36371077, 0.17985623],
                        [0.15896958, 0.11037514, 0.07244247],
                        [0.82099323, 0.09710128, 0.08136552],
                        [0.41426299, 0.0641475, 0.04442035],
                        [0.09394051, 0.5759465, 0.08729856],
                        [0.84640867, 0.69947928, 0.04568374],
                        [0.23789282, 0.934214, 0.04039037],
                        [0.82076712, 0.90884372, 0.07434012],
                        [0.09961493, 0.94530153, 0.04755969],
                        [0.88172021, 0.2724369, 0.04483477],
                        [0.9425836, 0.6339977, 0.04979664]])

    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i, coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any([is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA


def extract_city_area_information(train_x: np.ndarray, test_x: np.ndarray) -> (
        typing.Tuple)[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the city_area information from the training and test features.
    @param train_x: training features.
    @param test_x: test features.
    @return: tuple of (training features' 2D coordinates, training features' city_area information, test features' 2D
    coordinates, test features' city_area information)
    """

    train_x_2D = np.zeros((train_x.shape[0], 2), dtype=float)
    train_x_AREA = np.zeros((train_x.shape[0],), dtype=bool)
    test_x_2D = np.zeros((test_x.shape[0], 2), dtype=float)
    test_x_AREA = np.zeros((test_x.shape[0],), dtype=bool)

    train_x_2D[:, :2] = train_x[:, :2]
    train_x_AREA[:] = train_x[:, 2] == 1.

    test_x_2D[:, :2] = test_x[:, :2]
    test_x_AREA[:] = test_x[:, 2] == 1.

    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA


class ExactGPModel(ExactGP):
    """
    Class for GP Regressor.
    """

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel() + MaternKernel() + MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def change_kernel(self, kernel):
        self.covar_module = kernel


class Model(object):
    """
    Class for model for GP regression.
    """

    def __init__(self):
        # First model for area 0.
        self.likelihood_0 = None
        self.model_0 = None

        # Second model for area 1.
        self.likelihood_1 = None
        self.model_1 = None

        # Device.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def make_predictions(self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray) -> (
            typing.Tuple)[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        @param test_x_2D: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2).
        @param test_x_AREA: city_area info for every sample in a form of a bool array (NUM_SAMPLES,).
        @return: Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,), containing the predictions,
         the GP posterior mean, and the GP posterior stdandard deviation (in that order).
        """

        print('Predicting on test features.')

        # Create tensor.
        test_x_2D = torch.tensor(test_x_2D, dtype=torch.float32).to(self.device)
        test_x_AREA = torch.tensor(test_x_AREA, dtype=torch.float32).to(self.device)

        # Find indeces of samples in area 0 and 1.
        indices_test_0 = torch.nonzero(test_x_AREA == False).to(self.device)
        indices_test_1 = torch.nonzero(test_x_AREA == True).to(self.device)

        # Separate test data accordingly.
        test_x_2D_0 = test_x_2D[indices_test_0]
        test_x_2D_1 = test_x_2D[indices_test_1]

        # Make predictions for first model.
        self.model_0.to(self.device)
        self.likelihood_0.to(self.device)

        self.model_0.eval()
        self.likelihood_0.eval()

        with torch.no_grad():
            f_pred_0 = self.model_0(test_x_2D_0)
            gp_mean_0 = f_pred_0.mean.cpu().numpy()
            gp_std_0 = torch.sqrt(f_pred_0.variance).cpu().numpy()

        predictions_0 = gp_mean_0

        # Make predictions for second model.
        self.model_1.to(self.device)
        self.likelihood_1.to(self.device)

        self.model_1.eval()
        self.likelihood_1.eval()

        with torch.no_grad():
            f_pred_1 = self.model_1(test_x_2D_1)
            gp_mean_1 = f_pred_1.mean.cpu().numpy()
            gp_std_1 = torch.sqrt(f_pred_1.variance).cpu().numpy()

        predictions_1 = gp_mean_1

        for i in range(len(predictions_1)):
            predictions_1[i] += ADJ_FACTOR_AREA_1 * gp_std_1[i]

        # Create new arrays of predictions, mean and standard deviation with ordered elements, then return everything.
        predictions = np.zeros(test_x_2D.shape[0], dtype=float)
        gp_mean = np.zeros(test_x_2D.shape[0], dtype=float)
        gp_std = np.zeros(test_x_2D.shape[0], dtype=float)

        for i, index in enumerate(indices_test_0.tolist()):
            predictions[index] = predictions_0[i]
            gp_mean[index] = gp_mean_0[i]
            gp_std[index] = gp_std_0[i]

        for i, index in enumerate(indices_test_1.tolist()):
            predictions[index] = predictions_1[i]
            gp_mean[index] = gp_mean_1[i]
            gp_std[index] = gp_std_1[i]

        return predictions, gp_mean, gp_std

    def train_submodel(self, model, likelihood, x_train, y_train):
        """
        Train a model using negative marginal log-likelihood as objective.
        @param model: the model to train.
        @param likelihood: the likelihood function. For GPR, the Gaussian likelihood.
        @param x_train: the training samples, already sent to the appropriate device.
        @param y_train: the lables, already sent to the appropriate device.
        """

        # Send model to device.
        model.to(self.device)
        likelihood.to(self.device)

        # Set model and likelihhod for training.
        model.train()
        likelihood.train()

        # Use the adam optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=0.25)

        # "Loss" for GPs - the marginal log likelihood.
        mll = ExactMarginalLogLikelihood(likelihood, model)

        start_time = time.time()

        for i in range(TRAINING_ITERATIONS):
            # Zero gradients from previous iteration.
            optimizer.zero_grad()
            # Output from model
            output = model(x_train)
            # Calc loss and backprop gradients.
            loss = -mll(output, y_train)
            loss.backward()
            if (i + 1) % 50 == 0:
                print(
                    f"Iter {i + 1}/{TRAINING_ITERATIONS} - Loss: {loss.item()}  noise: {model.likelihood.noise.item()}")
            optimizer.step()

        end_time = time.time()
        print(f"Training took {end_time - start_time:.2f} seconds.")

    def fitting_model(self, train_y: np.ndarray, train_x_2D: np.ndarray,):
        """
        Fit model on the given training data.
        @param train_y: raining pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,).
        @param train_x_2D: training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2).
        """

        print("Start fitting the model.")

        # Get area id information. We cannot pass it as argument for compatibility issues
        # (the signature of the method cannot change).
        train_x_AREA = determine_city_area_idx(train_x_2D)
        train_x_AREA = train_x_AREA == 1.

        # Extract training data for area id 0.
        x_train_0 = train_x_2D[train_x_AREA == False]
        y_train_0 = train_y[train_x_AREA == False]

        # _, x_train_0, y_train_0 = cluster_data(x_train_0, y_train_0, k=NUM_SAMPLES, plot=False)

        # Create tensors.
        x_train_0 = torch.tensor(x_train_0, dtype=torch.float32).to(self.device)
        y_train_0 = torch.tensor(y_train_0, dtype=torch.float32).to(self.device)

        # Create likelihood and model.
        self.likelihood_0 = GaussianLikelihood()
        self.model_0 = ExactGPModel(x_train_0, y_train_0, self.likelihood_0)

        # Train model 0.
        print(f"Start training model for area 0...")
        self.train_submodel(self.model_0, self.likelihood_0, x_train_0, y_train_0)

        # Extract training data for area id 1.
        x_train_1 = train_x_2D[train_x_AREA == True]
        y_train_1 = train_y[train_x_AREA == True]

        # _, x_train_1, y_train_1 = cluster_data(x_train_1, y_train_1, k=NUM_SAMPLES, plot=False)

        # Create tensors.
        x_train_1 = torch.tensor(x_train_1, dtype=torch.float32).to(self.device)
        y_train_1 = torch.tensor(y_train_1, dtype=torch.float32).to(self.device)

        # Create likelihood and model.
        self.likelihood_1 = GaussianLikelihood()
        self.model_1 = ExactGPModel(x_train_1, y_train_1, self.likelihood_1)

        # Train model 1.
        print(f"Start training model for area 1...")
        self.train_submodel(self.model_1, self.likelihood_1, x_train_1, y_train_1)


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualize the predictions of a fitted model.
    @param model: fitted model to be visualized.
    @param output_dir: directory in which the visualizations will be stored.
    """

    print('Performing extended evaluation...')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs_2D, visualization_xs_AREA)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1.')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}.')

    plt.show()


def main():
    # Load the training dateset and test features.
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information.
    train_x_2D, _, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)

    # Fit the model.
    model = Model()
    model.fitting_model(train_y, train_x_2D)

    # Predict on the test features.
    predictions = model.make_predictions(test_x_2D, test_x_AREA)
    # print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
