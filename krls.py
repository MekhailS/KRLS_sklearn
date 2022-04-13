import numpy as np
from sklearn.base import BaseEstimator


class KRLS(BaseEstimator):
    def __init__(
            self,
            delta_threshold=0.001,
            gamma=0.8,
            kernel_type="RBF"
    ):
        self.gamma = gamma
        self.kernel_type = kernel_type
        kernels = {
            "RBF": lambda a, b: np.exp(-self.gamma * np.linalg.norm(a - b)**2)
        }
        assert kernel_type in kernels
        self.kernel = lambda a, b: kernels[self.kernel_type](a, b) + 1

        self.delta_threshold = delta_threshold

        self.is_initialized = False

        self.K = None
        self.K_inv = None
        self.A = None
        self.P = None
        self.K = None
        self.alpha = None
        self.D_set = None
        self.m = None

    def fit(self, X, y):
        assert len(X) == len(y)

        for (x_train, y_train) in zip(X, y):
            if not self.is_initialized:
                self._init_with_first_sample(x_train, y_train)
                continue

            self._update(x_train, y_train)

        return self

    def _init_with_first_sample(self, first_input, first_output):
        self.is_initialized = True

        k_11 = self.kernel(first_input, first_input)

        self.K = np.asarray([[k_11]])
        self.K_inv = np.asarray([[1.0 / k_11]])
        self.A = np.asarray([[1.0]])
        self.P = np.asarray([[1.0]])
        self.K = np.asarray([[1.0]])

        self.alpha = np.asarray([first_output / k_11])

        self.D_set = [first_input]

        self.m = 1

    def predict(self, X):
        return np.asarray([self._predict_on_one_sample(x) for x in X])

    def _predict_on_one_sample(self, new_input):
        h = np.array(
            [self.kernel(new_input, old_input) for old_input in self.D_set]
        ).reshape(1, len(self.D_set))
        return np.dot(h, self.alpha)

    def _update(self, new_input, expected):
        # h:= k_{t-1}(x_t)
        h = np.array([
            self.kernel(new_input, old_input) for old_input in self.D_set
        ]).reshape(len(self.D_set), 1)
        h_T = h.T

        k_tt = self.kernel(new_input, new_input)

        a = self.K_inv.dot(h)
        a_T = a.T

        assert(a_T.dot(h).size == 1)

        delta = np.abs(k_tt - a_T.dot(h)[0][0])

        if delta > self.delta_threshold:
            self.D_set.append(new_input)
            # Construct new
            new_K_size = len(a) + 1
            new_K_inv = np.zeros((new_K_size, new_K_size))
            new_K_inv[0:new_K_size - 1, 0:new_K_size - 1] = self.K_inv * delta + a.dot(a_T)
            new_K_inv[0:new_K_size - 1, new_K_size - 1:new_K_size] = -a
            new_K_inv[new_K_size - 1:new_K_size, 0:new_K_size - 1] = -a_T
            new_K_inv[new_K_size - 1][new_K_size - 1] = 1.0
            self.K_inv = new_K_inv / delta

            new_K = np.zeros((new_K_size, new_K_size))
            new_K[0:new_K_size - 1, 0:new_K_size - 1] = self.K
            new_K[0:new_K_size - 1, new_K_size - 1:new_K_size] = h
            new_K[new_K_size - 1:new_K_size, 0:new_K_size - 1] = h_T
            new_K[new_K_size - 1][new_K_size - 1] = k_tt
            self.K = new_K

            new_A = np.zeros(np.asarray(self.A.shape) + 1)
            new_A[0:-1, 0:-1] = self.A
            new_A[-1, -1] = 1.0
            self.A = new_A

            new_P = np.zeros(np.asarray(self.P.shape) + 1)
            new_P[0:-1, 0:-1] = self.P
            new_P[-1, -1] = 1.0
            self.P = new_P

            # _update alpha
            common_alpha_part = expected - h_T.dot(self.alpha) / delta
            assert common_alpha_part.size == 1

            new_alpha = np.zeros(np.asarray(self.alpha.shape) + 1)
            new_alpha[0:-1] = self.alpha - common_alpha_part
            new_alpha[-1] = common_alpha_part
            self.alpha = new_alpha

            self.m += 1

        else:
            q = self.P.dot(a) / (1.0 + (a_T.dot(self.P)).dot(a))
            new_P = self.P - \
                    (((self.P.dot(a)).dot(a_T)).dot(self.P)) \
                    / (1.0 + (a_T.dot(self.P)).dot(a))
            self.P = new_P

            new_alpha = self.alpha + (self.K_inv.dot(q)).dot(
                expected - h_T.dot(self.alpha)
            )
            self.alpha = new_alpha

            new_A = np.hstack((
                self.A.T, a
            )).T
            self.A = new_A