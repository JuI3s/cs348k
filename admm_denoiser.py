import numpy as np
import scipy as sp
from sksparse.cholmod import cholesky
from util import soft_threshold


class ADMMDenoiser:
    def __init__(self, img, rho=2, lmbda=30, verbose=True) -> None:
        self.img = img
        self.rho = rho
        self.lmbda = lmbda
        self.verbose = verbose

    def compute_diff(self):
        pass

    def compute_reg(self):
        pass

    def one_pass(self):
        pass

    def solve(self):
        pass


class ADMMDenoiserTV(ADMMDenoiser):
    def __init__(self, img, verbose) -> None:
        super().__init__(img, verbose)
        self.setup_params(img)
        self.img_flat = img.flatten()

        self.z = self.img_flat + np.random.rand(self.img_flat.shape[0])
        self.x = self.img_flat + np.random.rand(self.img_flat.shape[0])
        self.u = np.random.rand(self.img_flat.shape[0])

        self.img_shape = img.shape

    def setup_params(self, img):
        self.setup_F_and_factor(img.shape)

    def setup_F_and_factor(self, img_shape):
        # We cache the matrixes F, F.transpose(), and the Cholesky factor
        # of (Id + rho * F.transpose @ F)

        if self.verbose:
            print(f"Image shape in setting up ADMM denoiser: {img_shape}")

        img_flat_dim = img_shape[0] * img_shape[1]

        num_cols = img_shape[1]

        data = [2 if i % 2 == 0 else -1 for i in range(2 * img_flat_dim - 1)]
        data = data + [-1 for i in range(img_flat_dim - num_cols)]

        rows = [int(i / 2) for i in range(2 * img_flat_dim - 1)]
        rows = rows + [i for i in range(img_flat_dim - num_cols)]

        cols = [
            int(i / 2) if i % 2 == 0 else int(i / 2) + 1
            for i in range(2 * img_flat_dim - 1)
        ]
        cols = cols + [i + num_cols for i in range(img_flat_dim - num_cols)]

        self.F = sp.sparse.csr_matrix(
            (data, (rows, cols)), shape=(img_flat_dim, img_flat_dim)
        )
        self.Ft = self.F.transpose()

        FtF = self.F.transpose() @ self.F
        self.factor = cholesky(
            sp.sparse.identity(img_flat_dim) + self.rho * FtF,
        )

        if self.verbose:
            print(f"Set up F matrix:\n {self.F}\n")
            print(f"Set up Cholesky factor:\n {self.factor}\n")

    def compute_diff(self):
        self.x = self.factor(self.img_flat + self.rho * self.Ft @ (self.z - self.u))

    def compute_reg(self):
        self.z = soft_threshold(self.lmbda / self.rho, self.F @ self.x + self.u)

    def dual_update(self):
        self.u = self.u + self.F @ self.x - self.z

    def one_pass(self):
        self.compute_diff()
        self.compute_reg()
        self.dual_update()

    def solve(self):
        for _ in range(20):
            # for _ in range(100):
            self.one_pass()
        print(f"ADMM solutoin {self.x}")
        return self.x.reshape(self.img_shape)
