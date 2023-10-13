from ..ObjectiveFunc import ObjectiveVectorFunc
from ..encodings import ImageEncoding
import numpy as np

# from numba import jit
from skimage import metrics


class ImgApprox(ObjectiveVectorFunc):
    def __init__(self, img_dim, reference, opt="min", img_name=""):
        self.img_dim = tuple(img_dim) + (3,)
        self.size = img_dim[0] * img_dim[1] * 3
        self.reference = reference.resize((img_dim[0], img_dim[1]))
        self.reference = np.asarray(self.reference)[:, :, :3].astype(np.uint8)

        if img_name == "":
            name = "Image approximation"
        else:
            name = f'Approximating "{img_name}"'

        opt = "max"

        super().__init__(self.size, opt, 0, 256, name=name)

    def objective(self, solution):
        # return imgdistance(solution, self.reference)
        return (
            sum(
                metrics.structural_similarity(
                    solution[:, :, i], self.reference[:, :, i]
                )
                for i in range(3)
            )
            / 3
        )

    def repair_solution(self, solution):
        return np.clip(solution, 0, 255)


# @jit(nopython=True)
def imgdistance(img, reference):
    return np.sum((img - reference) ** 2)


class ImgStd(ObjectiveVectorFunc):
    def __init__(self, img_dim, opt="max"):
        self.size = img_dim[0] * img_dim[1] * 3

        if encoding is None:
            encoding = ImageEncoding(img_dim, color=True)

        super().__init__(self.size, opt, 0, 256, name="Image standard deviation")

    def objective(self, solution):
        solution_color = solution.reshape([3, -1])
        return solution_color.std(axis=1).max()

    def repair_solution(self, solution):
        return np.clip(solution, 0, 255).astype(np.uint8)


class ImgEntropy(ObjectiveVectorFunc):
    def __init__(self, img_dim, nbins=10, opt="min"):
        self.size = img_dim[0] * img_dim[1] * 3
        self.nbins = 10

        super().__init__(self.size, opt, 0, 256, name="Image entropy")

    def objective(self, solution):
        solution_channels = solution.reshape([3, -1])
        img_hists = [
            np.histogram(solution_channels[i], bins=np.linspace(0, 256, self.nbins))[0]
            for i in range(3)
        ]
        img_hists = np.array(img_hists) / solution_channels.shape[1]
        img_hists_no_zeros = img_hists
        img_hists_no_zeros[img_hists == 0] = 1
        return np.sum(-img_hists * np.log(img_hists_no_zeros))

    def repair_solution(self, solution):
        return np.clip(solution, 0, 255).astype(np.uint8)


class ImgExperimental(ObjectiveVectorFunc):
    def __init__(self, img_dim, reference, img_name, opt="min"):
        self.img_dim = tuple(img_dim) + (3,)
        self.size = img_dim[0] * img_dim[1] * 3
        self.reference = np.asarray(reference.resize([img_dim[0], img_dim[1]]))[
            :, :, :3
        ].astype(np.uint32)

        super().__init__(self.size, opt, 0, 256, name="Image approx and std")

    def objective(self, solution):
        dist = imgdistance(solution, self.reference)
        dist_norm = dist / (np.sqrt(self.size) * 255)

        solution_rounded = solution // 75
        solution_color = solution_rounded.reshape([3, -1])
        _, counts = np.unique(solution_color, axis=1, return_counts=True)
        freq = counts / self.size
        entropy = -(freq * np.log(freq)).sum()

        solution_color = solution.reshape([3, -1])
        dev = -solution_color.std(axis=1).max()

        return dist_norm + dev

    def repair_solution(self, solution):
        return np.clip(solution, 0, 255)

    def repair_speed(self, solution):
        return np.clip(solution, -255, 255)
