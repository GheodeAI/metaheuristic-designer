import sys
sys.path.append("../../..")

from PyMetaheuristics import *

import math
from numba import jit
import numpy as np
from PIL import Image
import cv2

class ImgExperimental(ObjectiveFunc):
    def __init__(self, img_dim, reference, opt="min"):
        self.size = img_dim[0]*img_dim[1]*3
        self.reference = np.asarray(reference.resize([img_dim[0], img_dim[1]]))[:,:,:3].flatten().astype(np.uint32)
        super().__init__(self.size, opt, "Image optimization function")
    
    def objective(self, solution):
        dist = imgdistance(solution, self.reference)
        dist_norm = dist/(np.sqrt(self.size)*255)

        solution_rounded = solution//75
        solution_color = solution_rounded.reshape([3,-1])
        _, counts = np.unique(solution_color, axis=1, return_counts=True)
        freq = counts/self.size
        entropy = -(freq*np.log(freq)).sum()

        solution_color = solution.reshape([3,-1])
        dev = solution_color.std(axis=1).max()

        return dist_norm*dev

    def random_solution(self):
        return np.random.randint(0,256,size=64*64*3)
    
    def repair_solution(self, solution):
        return np.clip(solution, 0, 255)
    
    def repair_speed(self, solution):
        return np.clip(solution, -255, 255)

class ImgApprox(ObjectiveFunc):
    def __init__(self, img_dim, reference, opt="min", img_name=""):
        self.size = img_dim[0]*img_dim[1]*3
        self.reference = reference.resize((img_dim[0],img_dim[1]))
        self.reference = np.asarray(self.reference)[:, :, :3].flatten().astype(np.uint8)

        if img_name == "":
            name = "Image approximation"
        else:
            name = f"Approximating \"{img_name}\""

        super().__init__(self.size, opt, name)
    
    def objective(self, solution):
        #return imgdistance(solution, self.reference)
        return np.sum((solution-self.reference)**2)

    def random_solution(self):
        return np.random.randint(0, 256, size=self.size)
    
    def repair_solution(self, solution):
        return np.clip(solution, 0, 255)


class ImgStd(ObjectiveFunc):
    def __init__(self, img_dim, opt="max"):
        self.size = img_dim[0]*img_dim[1]*3
        super().__init__(self.size, opt, "Image standard deviation")
    
    def objective(self, solution):
        # The distance between a white and a black image is of sqrt(N*M*3)*255
        solution_color = solution.reshape([3,-1])
        return solution_color.std(axis=1).max()

    def random_solution(self):
        return np.random.randint(0, 256, size=self.size)
    
    def repair_solution(self, solution):
        return np.clip(solution, 0, 255).astype(np.uint8)

class ImgEntropy(ObjectiveFunc):
    def __init__(self, img_dim, nbins=10, opt="min"):
        self.size = img_dim[0]*img_dim[1]*3
        self.nbins = 10
        print("a")
        super().__init__(self.size, opt, "Image entropy")
    
    def objective(self, solution):
        solution_channels = solution.reshape([3, -1])
        img_hists = [np.histogram(solution_channels[i], bins=np.linspace(0,256,self.nbins))[0] for i in range(3)]
        img_hists = np.array(img_hists) / solution_channels.shape[1]
        img_hists_no_zeros = img_hists
        img_hists_no_zeros[img_hists==0] = 1
        return np.sum(-img_hists*np.log(img_hists_no_zeros))

    def random_solution(self):
        return np.random.randint(0, 256, size=self.size)
    
    def repair_solution(self, solution):
        return np.clip(solution, 0, 255).astype(np.uint8)

# class ImgEntropy(ObjectiveFunc):
#     def __init__(self, img_dim, opt="min"):
#         self.size = img_dim[0]*img_dim[1]*3
#         super().__init__(self.size, opt, "Image entropy")
    
#     def objective(self, solution):
#         # The distance between a white and a black image is of sqrt(N*M*3)*255
#         solution_rounded = solution//75
#         solution_color = solution_rounded.reshape([3,-1])
#         _, counts = np.unique(solution_color, axis=1, return_counts=True)
#         freq = counts/self.size
#         return -(freq*np.log(freq)).sum()

#     def random_solution(self):
#         return np.random.randint(0,256,size=64*64*3)
    
#     def repair_solution(self, solution):
#         return np.clip(solution, 0, 255).astype(np.int32)

@jit(nopython=True)
def imgdistance(img, reference):
    return ((img-reference)**2).sum()