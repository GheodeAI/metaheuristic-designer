from ..ObjectiveFunc import ObjectiveVectorFunc
from ..encodings import ImageEncoding
import numpy as np

# from numba import jit
from skimage import metrics


class ImgApprox(ObjectiveVectorFunc):
    def __init__(self, img_dim, reference, mode=None, img_name="", diff_func="MSE", name=None):
        self.img_dim = tuple(img_dim) + (3,)
        self.size = img_dim[0] * img_dim[1] * 3
        self.reference = reference.resize((img_dim[0], img_dim[1]))
        self.reference = np.asarray(self.reference)[:, :, :3].astype(np.uint8)

        if name is None:
            if img_name == "":
                name = "Image approximation"
            else:
                name = f'Approximating "{img_name}"'

        self.diff_func = diff_func
        if mode is None:
            if diff_func in ["MSE", "MAE"]:
                mode = "min"
            else:
                mode = "max"

        super().__init__(self.size, mode=mode, low_lim=0, up_lim=256, name=name, vectorized=True)

    def objective(self, solutions):
        error = np.zeros(solutions.shape[0])
        image_size = np.prod(solutions.shape[1:])
        match self.diff_func:
            case "MSE":
                error = np.astype(np.sum((solutions - self.reference) ** 2, axis=(1, 2, 3)) / image_size, float)
            case "MAE":
                error = np.astype(np.sum(np.abs(solutions - self.reference), axis=(1, 2, 3)) / image_size, float)
            case "SSIM":
                for idx, s in enumerate(solutions):
                    for s_ch, ref_ch in zip(s.transpose((2, 0, 1)), self.reference.transpose((2, 0, 1))):
                        error[idx] += metrics.structural_similarity(s_ch, ref_ch)
                    error[idx] /= 3
            case "NMI":
                for idx, s in enumerate(solutions):
                    for s_ch, ref_ch in zip(s.transpose((2, 0, 1)), self.reference.transpose((2, 0, 1))):
                        error[idx] += metrics.normalized_mutual_information(s_ch, ref_ch, bins=256)
                    error[idx] /= 3

        return error

    def repair_solution(self, solution):
        return np.clip(solution, 0, 255)

    def repair_speed(self, solution):
        return np.clip(solution, -100, 100)


class ImgStd(ObjectiveVectorFunc):
    def __init__(self, img_dim, mode=None):
        self.size = img_dim[0] * img_dim[1] * 3
        if mode is None:
            mode = "max"

        super().__init__(self.size, mode=mode, low_lim=0, up_lim=256, name="Image standard deviation")

    def objective(self, solution):
        solution_color = solution.reshape([3, -1])
        return solution_color.std(axis=1).mean()

    def repair_solution(self, solution):
        return np.clip(solution, 0, 255)

    def repair_speed(self, solution):
        return np.clip(solution, -100, 100)


class ImgEntropy(ObjectiveVectorFunc):
    def __init__(self, img_dim, nbins=10, mode=None):
        self.size = img_dim[0] * img_dim[1] * 3
        self.nbins = nbins
        if mode is None:
            mode = "max"

        super().__init__(self.size, mode=mode, low_lim=0, up_lim=256, name="Image entropy")

    def objective(self, solution):
        solution_channels = solution.reshape([3, -1])
        img_hists = [np.histogram(solution_channels[i], bins=np.linspace(0, 256, self.nbins))[0] for i in range(3)]
        img_hists = np.array(img_hists) / solution_channels.shape[1]
        img_hists_no_zeros = img_hists
        img_hists_no_zeros[img_hists == 0] = 1
        return np.sum(-img_hists * np.log(img_hists_no_zeros))

    def repair_solution(self, solution):
        return np.clip(solution, 0, 255)

    def repair_speed(self, solution):
        return np.clip(solution, -100, 100)


class ImgExperimental(ObjectiveVectorFunc):
    def __init__(self, img_dim, reference, img_name, mode=None):
        self.img_dim = tuple(img_dim) + (3,)
        self.size = img_dim[0] * img_dim[1] * 3
        self.reference = np.asarray(reference.resize([img_dim[0], img_dim[1]]))[:, :, :3].astype(np.uint32)
        if mode is None:
            mode = "max"

        super().__init__(self.size, mode=mode, low_lim=0, up_lim=256, name="Image approx and std")

    def objective(self, solution):
        dist = imgdistance_mse(solution, self.reference)
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
        return np.clip(solution, -100, 100)
