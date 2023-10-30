from metaheuristic_designer import (
    ObjectiveFunc,
    ParamScheduler,
    Individual,
)
from metaheuristic_designer.algorithms import GeneralAlgorithm, MemeticAlgorithm
from metaheuristic_designer.operators import OperatorReal, OperatorInt, OperatorBinary
from metaheuristic_designer.strategies import *
from metaheuristic_designer.initializers import *
from metaheuristic_designer.selectionMethods import ParentSelection, SurvivorSelection
from metaheuristic_designer.encodings import ImageEncoding
from metaheuristic_designer.benchmarks import *

import pygame
import time
import numpy as np
import cv2
import os
from copy import deepcopy
from PIL import Image
import skimage.filters

# import matplotlib
# matplotlib.use("Gtk3Agg")

import argparse


class ImageBlurEncoding(ImageEncoding):
    def __init__(self, shape, color=True):
        super().__init__(shape, color=True)

    def decode(self, genotype: np.ndarray) -> np.ndarray:
        image_matrix = np.reshape(genotype, self.shape)
        return skimage.filters.gaussian(image_matrix, channel_axis=-1).astype(np.uint8)


def render(image, display_dim, src):
    texture = cv2.resize(
        image.transpose([1, 0, 2]),
        (display_dim[1], display_dim[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    pygame.surfarray.blit_array(src, texture)
    pygame.display.flip()


def save_to_image(image, img_name="result.png"):
    if not os.path.exists("./examples/results/"):
        os.makedirs("./examples/results/")
    filename = "./examples/results/" + img_name
    Image.fromarray(image.astype(np.uint8)).save(filename)


def run_algorithm(alg_name, img_file_name, memetic):
    params = {
        # General
        "stop_cond": "time_limit",
        "time_limit": 300.0,
        "ngen": 1000,
        "neval": 3e5,
        "fit_target": 0,
        "verbose": True,
        # "verbose": False,
        "v_timer": 0.5,
    }

    display = True
    display_dim = [1200, 600]
    image_shape = [64, 64]

    if display:
        pygame.init()
        src = pygame.display.set_mode(display_dim)
        pygame.display.set_caption("Evo graphics")

    reference_img = Image.open(img_file_name)
    img_name = img_file_name.split("/")[-1]
    img_name = img_name.split(".")[0]
    objfunc = ImgApprox(image_shape, reference_img, img_name=img_name)
    # objfunc = ImgEntropy(image_shape, 256)
    # objfunc = ImgExperimental(image_shape, reference_img, img_name=img_name)

    objfunc.name = "Image debluring"

    deblured_encoding = ImageEncoding(image_shape, color=True)
    encoding = ImageBlurEncoding(image_shape, color=True)
    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize,
        objfunc.low_lim,
        objfunc.up_lim,
        encoding=encoding,
        pop_size=100,
    )

    init_population = [
        Individual(
            objfunc,
            deblured_encoding.encode(np.asarray(reference_img)[:, :, :3].flatten())
            + np.random.normal(0, 2, np.asarray(reference_img)[:, :, :3].size),
            encoding=encoding,
        )
        for i in range(100)
    ]
    pop_initializer = DirectInitializer(
        pop_initializer, init_population, encoding=encoding
    )

    mutation_op = OperatorReal("MutRand", {"method": "Cauchy", "F": 4, "N": 2})
    cross_op = OperatorReal("Multicross", {"Nindiv": 4})
    parent_sel_op = ParentSelection("Best", {"amount": 15})
    selection_op = SurvivorSelection("Elitism", {"amount": 10})

    mem_select = ParentSelection("Best", {"amount": 5})
    neihbourhood_op = OperatorReal("RandNoise", {"method": "Cauchy", "F": 0.0002})
    local_search = LocalSearch(pop_initializer, neihbourhood_op, params={"iters": 10})

    if alg_name == "HillClimb":
        pop_initializer.pop_size = 1
        search_strat = HillClimb(pop_initializer, mutation_op)
    elif alg_name == "LocalSearch":
        pop_initializer.pop_size = 1
        search_strat = LocalSearch(pop_initializer, mutation_op, {"iters": 20})
    elif alg_name == "SA":
        pop_initializer.pop_size = 1
        search_strat = SA(
            pop_initializer, mutation_op, {"iter": 100, "temp_init": 2, "alpha": 0.998}
        )
    elif alg_name == "ES":
        search_strat = ES(
            pop_initializer,
            mutation_op,
            cross_op,
            parent_sel_op,
            selection_op,
            {"offspringSize": 150},
        )
    elif alg_name == "GA":
        search_strat = GA(
            pop_initializer,
            mutation_op,
            cross_op,
            parent_sel_op,
            selection_op,
            {"popSize": 100, "pcross": 0.8, "pmut": 0.4},
        )
    elif alg_name == "HS":
        search_strat = HS(pop_initializer, {"HMCR": 0.8, "BW": 0.5, "PAR": 0.2})
    elif alg_name == "DE":
        search_strat = DE(
            pop_initializer, OperatorReal("DE/best/1", {"F": 0.8, "Cr": 0.8})
        )
    elif alg_name == "PSO":
        search_strat = PSO(pop_initializer, {"w": 0.7, "c1": 1.5, "c2": 1.5})
    elif alg_name == "RandomSearch":
        pop_initializer.pop_size = 1
        search_strat = RandomSearch(pop_initializer)
    elif alg_name == "NoSearch":
        pop_initializer.pop_size = 1
        search_strat = NoSearch(pop_initializer)
    else:
        print(f'Error: Algorithm "{alg_name}" doesn\'t exist.')
        exit()

    if memetic:
        ls_pop_init = UniformVectorInitializer(
            objfunc.vecsize,
            objfunc.low_lim,
            objfunc.up_lim,
            encoding=encoding,
            pop_size=100,
        )
        local_search = LocalSearch(
            ls_pop_init,
            OperatorInt("MutRand", {"method": "Uniform", "Low": -3, "Up": -3, "N": 3}),
            params={"iters": 10},
        )
        alg = MemeticAlgorithm(
            objfunc,
            search_strat,
            local_search,
            ParentSelection("Best", {"amount": 10}),
            params,
        )
    else:
        alg = GeneralAlgorithm(objfunc, search_strat, params)

    # Optimize with display of image
    real_time_start = time.time()
    cpu_time_start = time.process_time()
    display_timer = time.time()

    alg.initialize()

    while not alg.ended:
        # process GUI events and reset screen
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit(0)
            src.fill("#000000")

        alg.step(time_start=real_time_start)

        alg.update(real_time_start, cpu_time_start)

        if alg.verbose and time.time() - display_timer > alg.v_timer:
            alg.step_info(real_time_start)
            display_timer = time.time()

        if display:
            img_flat = alg.best_solution()[0]
            image_blur = encoding.decode(img_flat)
            image_orig = deblured_encoding.decode(img_flat)
            full_texture = np.hstack([image_orig, image_blur])
            # print(full_texture.shape)
            # full_texture = image_orig
            # full_texture = image_blur
            render(full_texture, display_dim, src)
            pygame.display.update()

    alg.real_time_spent = time.time() - real_time_start
    alg.time_spent = time.process_time() - cpu_time_start
    img_flat = alg.best_solution()[0]
    image = img_flat.reshape(image_shape + [3])
    if display:
        img_flat = alg.best_solution()[0]
        image_blur = encoding.decode(img_flat)
        image_orig = deblured_encoding.decode(img_flat)
        full_texture = np.hstack([image_orig, image_blur])
        # print(full_texture.shape)
        # full_texture = image_orig
        # full_texture = image_blur
        render(full_texture, display_dim, src)
    alg.display_report(show_plots=True)
    save_to_image(
        image, f"{img_name}_{image_shape[0]}x{image_shape[1]}_deblured_{alg_name}.png"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest="alg", help="Specify an algorithm")
    parser.add_argument(
        "-i", "--image", dest="img", help="Specify an image as reference"
    )
    parser.add_argument(
        "-m", "--memetic", dest="mem", action="store_true", help="Specify an algorithm"
    )
    args = parser.parse_args()

    algorithm_name = "SA"
    img_file_name = "data/images/cat_blurry.png"
    mem = False

    if args.alg:
        algorithm_name = args.alg

    if args.img:
        img_file_name = args.img

    if args.mem:
        mem = True

    run_algorithm(alg_name=algorithm_name, img_file_name=img_file_name, memetic=mem)


if __name__ == "__main__":
    main()
