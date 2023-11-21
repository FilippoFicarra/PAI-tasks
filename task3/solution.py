"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm



# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.

class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        
        np.random.seed(42)
        
        sigma_f = 0.15
        self.kernel_f = Matern(nu=2.5, length_scale=0.5)
        self.f = GaussianProcessRegressor(kernel=self.kernel_f, random_state=42, alpha=sigma_f**2)
        
        # Kernel and prior mean setup for v
        sigma_v = 0.0001
        self.kernel_v = Matern(nu=2.5, length_scale=0.5) + ConstantKernel(4)
        self.v = GaussianProcessRegressor(kernel=self.kernel_v, random_state=42, alpha=sigma_v**2)
        
        # X = np.linspace(DOMAIN[0][0], DOMAIN[0][1], 100)[:, None]
        # y = np.ones(X.shape[0]) * 4
        # self.v.fit(X, y)
                
        # Data storage
        self.x = np.array([])
        self.f_x = np.array([])
        self.v_x = np.array([])
        
        self.sa = 4.0
    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        # if self.x.size == 0:
        #     x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
        #     c_val = np.vectorize(self.v)(x_domain)
        #     x_valid = x_domain[c_val < SAFETY_THRESHOLD]
        #     np.random.seed(0)
        #     np.random.shuffle(x_valid)
        #     x = x_valid[0]
        #     return x
        # else:
        x = self.optimize_acquisition_function()

        while np.abs(x-self.x[0]) <= 0.1:
            rand = np.random.uniform(-0.2, 0.2)
            new_x = self.optimize_acquisition_function() + rand
            x = np.clip(new_x , DOMAIN[0,0], DOMAIN[0,1])
        return x

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.

        mean_f, std_f = self.f.predict(x, return_std=True)
        mean_v, _ = self.v.predict(x, return_std=True)
        

        # EI
        # best_f = np.max(self.f_x)
        # delta = (best_f - mean_f)
        # z_f = delta / std_f
        # ei_f = delta * norm.cdf(z_f) + std_f * norm.pdf(z_f)
        
        # UCB
        # beta = 0.3
        # ei_f = mean_f + beta * std_f
        
        # LCB
        beta = 0.2
        ei_f = mean_f - beta * std_f
        
        # PI
        # best_f = np.max(self.f_x)
        
        # z_f = (mean_f - best_f) / std_f
        # ei_f = norm.cdf(z_f)
        
        # p = norm.cdf((self.sa - mean_v) / std_v)
        
        
        penalty = np.maximum(mean_v - self.sa, 0)
        lambda_param = 2.0
        
        return ei_f - lambda_param * penalty


    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.

        self.x = np.vstack((self.x, x)) if self.x.size else np.array([x])
        self.f_x = np.vstack((self.f_x, f)) if self.f_x.size else np.array([f])
        self.v_x = np.vstack((self.v_x, v)) if self.v_x.size else np.array([v])
        
        self.f.fit(self.x, self.f_x)
        self.v.fit(self.x, self.v_x)
        
    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.

        prediction = np.copy(self.f_x)
        prediction[np.where(self.v_x >= self.sa)] = -np.inf
        return self.x[np.argmax(prediction)]
        # search_range = np.linspace(DOMAIN[0][0], DOMAIN[0][1], 100000)
        # mean_f, _ = self.f.predict(search_range.reshape(-1, 1), return_std=True)
        # mean_v, _ = self.v.predict(search_range.reshape(-1, 1), return_std=True)
        # mean_f[np.where(mean_v >= self.sa)] = -np.inf
        # max_index = np.argmax(mean_f)
        # return search_range[max_index]
        
    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
