import numpy as np
import scipy as sp
from sksparse.cholmod import cholesky


class ADMMDenoiser:

    def __init__(self, img, rho=2, verbose=True) -> None:
        self.img = img
        self.rho = rho
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

    def setup_params(self, img):
        self.setup_F_and_factor(img.shape)

    def setup_F_and_factor(self, img_shape):
        # We cache the matrixes F, F.transpose(), and the Cholesky factor
        # of (Id + rho * F.transpose @ F)

        if self.verbose:
            print(f"Image shape in setting up ADMM denoiser: {img_shape}")

        img_flat_dim = img_shape[0] * img_shape[1]
        data = [1 if i % 2 == 0 else -1 for i in range(2 * img_flat_dim - 1)]
        rows = [int(i / 2) for i in range(2 * img_flat_dim - 1)]
        cols = [
            int(i / 2) if i % 2 == 0 else int(i / 2) + 1
            for i in range(2 * img_flat_dim - 1)
        ]

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
        pass

    def compute_reg(self):
        pass

    def solve(self):
        for _ in range(10):
            self.one_pass()
