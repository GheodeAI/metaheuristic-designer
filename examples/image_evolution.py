from metaheuristic_designer import ObjectiveFunc, ParamScheduler
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
from itertools import product

# import matplotlib
# matplotlib.use("Gtk3Agg")

import argparse


def render(image, display_dim, src):
    texture = cv2.resize(image.transpose([1, 0, 2]), display_dim, interpolation=cv2.INTER_NEAREST)
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
        "stop_cond": "convergence or time_limit",
        "progress_metric": "time_limit",
        "time_limit": 500.0,
        "cpu_time_limit": 120.0,
        "ngen": 1000,
        "neval": 3e5,
        "fit_target": 0,
        #"patience": 50,
        "patience": 200,
        "verbose": True,
        "v_timer": 0.5,
        # Parallel
        "parallel": False,
        "threads": 8,
    }

    display = True
    display_dim = [600, 600]
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

    encoding = ImageEncoding(image_shape, color=True)
    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize,
        objfunc.low_lim,
        objfunc.up_lim,
        pop_size=100,
        encoding=encoding,
    )

    mutation_op = OperatorReal("MutNoise", {"distrib": "Uniform", "min": -20, "max": 20, "N": 15})
    cross_op = OperatorReal("Multipoint")

    op_list = [
        OperatorReal("Multipoint"),
        OperatorReal("MutRand", {"distrib": "Cauchy", "F": 5, "N": 10}, name="MutCauchy"),
        OperatorReal("MutRand", {"distrib": "Gauss", "F": 5, "N": 10}, name="MutGauss"),
        OperatorReal(
            "MutSample",
            {"distrib": "Uniform", "min": 0, "max": 256, "N": 10},
            name="MutUniform",
        ),
    ]

    # n_list = np.flip(np.logspace(0, 12, base=2, num=1000))
    # n_list = np.logspace(4, 12, base=2, num=200)
    n_list = np.logspace(5, 12, base=2, num=200)

    neighborhood_structures = [
        OperatorReal(
            "MutNoise",
            {"distrib": "Uniform", "min": -10, "max": 10, "N": n},
            name=f"UniformSample(N={n:0.0f})",
        )
        for n in n_list
    ]

    parent_sel_op = ParentSelection("Best", {"amount": 15})
    selection_op = SurvivorSelection("Elitism", {"amount": 10})

    mem_select = ParentSelection("Best", {"amount": 5})
    neihbourhood_op = OperatorInt("MutRand", {"distrib": "Uniform", "min": -10, "max": -10, "N": 3})
    local_search = LocalSearch(pop_initializer, neihbourhood_op, params={"iters": 10})

    if alg_name == "HillClimb":
        pop_initializer.pop_size = 1
        search_strat = HillClimb(pop_initializer, mutation_op)
    elif alg_name == "LocalSearch":
        pop_initializer.pop_size = 1
        search_strat = LocalSearch(pop_initializer, mutation_op, params={"iters": 40})
    elif alg_name == "SA":
        pop_initializer.pop_size = 1
        search_strat = SA(pop_initializer, mutation_op, {"iter": 100, "temp_init": 1, "alpha": 0.997})
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
            {"pcross": 0.8, "pmut": 0.1},
        )
    elif alg_name == "HS":
        search_strat = HS(pop_initializer, {"HMCR": 0.9, "BW": 5, "PAR": 0.8})
    elif alg_name == "DE":
        search_strat = DE(pop_initializer, OperatorReal("DE/best/1", {"F": 0.8, "Cr": 0.8}))
    elif alg_name == "PSO":
        search_strat = PSO(pop_initializer, {"w": 0.7, "c1": 1.5, "c2": 1.5})
    elif alg_name == "CRO":
        search_strat = CRO(
            pop_initializer,
            mutation_op,
            cross_op,
            {"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
        )
    elif alg_name == "CRO_SL":
        search_strat = CRO_SL(
            pop_initializer,
            op_list,
            {"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
        )
    elif alg_name == "PCRO_SL":
        search_strat = PCRO_SL(
            pop_initializer,
            op_list,
            {"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
        )
    elif alg_name == "DPCRO_SL":
        search_strat_params = {
            "rho": 0.6,
            "Fb": 0.95,
            "Fd": 0.1,
            "Pd": 0.9,
            "attempts": 3,
            "group_subs": True,
            "dyn_method": "success",
            "dyn_metric": "best",
            "dyn_steps": 75,
            "prob_amp": 0.1,
        }
        search_strat = DPCRO_SL(pop_initializer, op_list, search_strat_params)
    elif alg_name == "RVNS":
        search_strat = RVNS(pop_initializer, neighborhood_structures)
    elif alg_name == "VND":
        search_strat = VND(pop_initializer, neighborhood_structures)
    elif alg_name == "VNS":
        local_search = LocalSearch(pop_initializer, params={"iters": 10})
        search_strat = VNS(
            pop_initializer,
            neighborhood_structures,
            local_search,
            params={"iters": 200, "nchange": "pipe"},
        )
        # local_search = HillClimb(pop_initializer)
        # search_strat = VNS(pop_initializer, neighborhood_structures, local_search, params={"iters": 500})
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
        alg = MemeticAlgorithm(objfunc, search_strat, local_search, mem_select, params=params)
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
            render(encoding.decode(img_flat), display_dim, src)
            pygame.display.update()

    alg.real_time_spent = time.time() - real_time_start
    alg.cpu_time_spent = time.process_time() - cpu_time_start
    img_flat = alg.best_solution()[0]
    image = img_flat.reshape(image_shape + [3])
    if display:
        render(image, display_dim, src)
    alg.display_report(show_plots=True)
    save_to_image(image, f"{img_name}_{image_shape[0]}x{image_shape[1]}_{alg_name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest="alg", help="Specify an algorithm")
    parser.add_argument("-i", "--image", dest="img", help="Specify an image as reference")
    parser.add_argument("-m", "--memetic", dest="mem", action="store_true", help="Specify an algorithm")
    args = parser.parse_args()

    algorithm_name = "SA"
    img_file_name = "data/images/cat.png"
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
