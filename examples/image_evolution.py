from metaheuristic_designer import ObjectiveFunc, ParamScheduler
from metaheuristic_designer.algorithms import GeneralAlgorithm, MemeticAlgorithm
from metaheuristic_designer.operators import OperatorVector
from metaheuristic_designer.operators import OperatorVector
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


def run_algorithm(alg_name, img_file_name, memetic, objfunc_name, mode, img_size, display):
    params = {
        # General
        # "stop_cond": "convergence or time_limit",
        "stop_cond": "time_limit",
        "progress_metric": "time_limit",
        "time_limit": 500.0,
        "patience": 1000,
        "verbose": True,
        "v_timer": 0.5,

        # Parallel
        "parallel": False,
        "threads": 8,
    }

    # Window size
    display_dim = [600, 600]

    # Image size
    image_shape = tuple(map(int, img_size.split(',')))

    # Initialize display
    if display:
        pygame.init()
        src = pygame.display.set_mode(display_dim)
        pygame.display.set_caption("Evo graphics")


    # Initialize objective function
    if objfunc_name is None or objfunc_name in ["MSE", "MAE", "SSIM", "NMI"]:

        # Load reference image
        reference_img = Image.open(img_file_name)
        img_name = img_file_name.split("/")[-1]
        img_name = img_name.split(".")[0]

        objfunc = ImgApprox(image_shape, reference_img, img_name=img_name, diff_func=objfunc_name, mode=mode)
    elif objfunc_name == "ENTROPY":
        objfunc = ImgEntropy(image_shape, 256, mode=mode)
    elif objfunc_name == "STD":
        objfunc = ImgStd(image_shape, mode=mode)

    # Prepare algorithm
    encoding = ImageEncoding(image_shape, color=True)
    pop_initializer = UniformVectorInitializer(
        objfunc.vecsize,
        objfunc.low_lim,
        objfunc.up_lim,
        pop_size=500,
        encoding=encoding,
    )

    # mutation_op = OperatorVector("MutNoise", {"distrib": "Uniform", "min": -20, "max": 20, "N": 15})
    mutation_op = OperatorVector("MutNoise", {"distrib": "Normal", "F": 10, "N": 5})
    cross_op = OperatorVector("Multipoint")

    op_list = [
        OperatorVector("Multipoint"),
        OperatorVector("MutRand", {"distrib": "Cauchy", "F": 5, "N": 10}, name="MutCauchy"),
        OperatorVector("MutRand", {"distrib": "Gauss", "F": 5, "N": 10}, name="MutGauss"),
    ]

    # n_list = np.flip(np.logspace(0, 12, base=2, num=1000))
    # n_list = np.logspace(4, 12, base=2, num=200)
    n_list = np.logspace(5, 12, base=2, num=200)

    neighborhood_structures = [
        OperatorVector(
            "MutNoise",
            {"distrib": "Uniform", "min": -10, "max": 10, "N": n},
            name=f"UniformSample(N={n:0.0f})",
        )
        for n in n_list
    ]

    parent_sel_op = ParentSelection("Best", {"amount": 15})
    selection_op = SurvivorSelection("Elitism", {"amount": 10})

    mem_select = ParentSelection("Best", {"amount": 5})
    neihbourhood_op = OperatorVector("MutRand", {"distrib": "Uniform", "min": -10, "max": -10, "N": 3})
    local_search = LocalSearch(pop_initializer, neihbourhood_op, params={"iters": 10})

    if alg_name == "HillClimb":
        pop_initializer.pop_size = 1
        search_strat = HillClimb(pop_initializer, mutation_op)
    elif alg_name == "LocalSearch":
        pop_initializer.pop_size = 1
        search_strat = LocalSearch(pop_initializer, mutation_op, params={"iters": 500})
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
        search_strat = DE(pop_initializer, OperatorVector("DE/best/1", {"F": 0.8, "Cr": 0.8}))
    elif alg_name == "PSO":
        search_strat = PSO(pop_initializer, {"w": 0.7, "c1": 1.5, "c2": 1.5})
    elif alg_name == "BinomialUMDA":
        search_strat = BinomialUMDA(pop_initializer, parent_sel_op, selection_op, params={"n": 256, "noise": 0})
    elif alg_name == "BinomialPBIL":
        search_strat = BinomialPBIL(pop_initializer, parent_sel_op, selection_op, params={"n": 256, "lr": 0.5, "noise": 0})
    elif alg_name == "GaussianUMDA":
        search_strat = GaussianUMDA(pop_initializer, parent_sel_op, selection_op, params={"scale": 7, "noise": 1})
    elif alg_name == "GaussianPBIL":
        search_strat = GaussianPBIL(pop_initializer, parent_sel_op, selection_op, params={"scale": 5, "lr": 0.5, "noise": 0.75})
    elif alg_name == "CrossEntropy":
        search_strat = CrossEntropyMethod(pop_initializer)
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
        pop_initializer.pop_size = 1
        local_search = LocalSearch(pop_initializer, mutation_op, params={"iters": 200})
        search_strat = VNS(
            initializer=pop_initializer,
            op_list=neighborhood_structures,
            local_search=local_search,
            params={"nchange": "seq"},
            inner_loop_params={
                "stop_cond": "convergence",
                "patience": 100,
                "verbose": params['verbose'],
                "v_timer": params['v_timer'],
            },
        )
    elif alg_name == "GVNS":
        pop_initializer.pop_size = 1
        search_strat = VNS(
            pop_initializer,
            neighborhood_structures,
            local_search,
            params={"iters": 200, "nchange": "pipe"},
        )
    elif alg_name == "GVNS":
        pop_initializer.pop_size = 1
        local_search = VND(pop_initializer, neighborhood_structures, params={"nchange": "cyclic"})
        search_strat = VNS(
            pop_initializer,
            neighborhood_structures,
            local_search,
            params={"iters": 200, "nchange": "pipe"},
        )
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
    alg.update(real_time_start, cpu_time_start, pass_step=False)

    while not alg.ended:
        # process GUI events and reset screen
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit(0)
            src.fill("#000000")

        alg.step(real_time_start)
        alg.update(real_time_start, cpu_time_start)

        if alg.verbose and time.time() - display_timer > alg.v_timer:
            alg.step_info(real_time_start)
            display_timer = time.time()

        if display:
            image, _ = alg.best_solution(decoded=True)
            render(image, display_dim, src)
            pygame.display.update()

    alg.real_time_spent = time.time() - real_time_start
    alg.cpu_time_spent = time.process_time() - cpu_time_start

    image, _ = alg.best_solution(decoded=True)
    if display:
        render(image, display_dim, src)
    alg.display_report(show_plots=True)


    memetic_str = "M-" if memetic else ""
    if objfunc_name is None or objfunc_name in ["MSE", "MAE", "SSIM", "NMI"]:
        out_img_name = f"{img_name}_{objfunc_name}_{image_shape[0]}x{image_shape[1]}_{memetic_str}{alg_name}.png"
    else:
        out_img_name = f"{objfunc_name}_{image_shape[0]}x{image_shape[1]}_{memetic_str}{alg_name}.png"

    save_to_image(image, out_img_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest="algorithm", help="Specify an algorithm.", default="SA")
    parser.add_argument("-i", "--image", dest="img", help="Specify an image as reference.", default="data/images/cat.png")
    parser.add_argument("-s", "--img_size", dest="img_size", help="Specify an image as reference.", default="32,32")
    parser.add_argument("-m", "--memetic", dest="mem", action="store_true", help="Use a memetic algorithm with the chosen strategy.")
    parser.add_argument("--hide", dest="hide", action="store_true", help="Do not display the result each generation.")
    parser.add_argument("--mode", dest="mode", help="Whether to minimize or maximize ('min' or 'max').")
    parser.add_argument("-o", "--objfunc", dest="objfunc", help="Which objective function to use ('MSE', 'MAE', 'STS', 'STD', 'Entropy').", default="mse")
    args = parser.parse_args()
    
    print(args.hide)

    run_algorithm(
        alg_name=args.algorithm,
        img_file_name=args.img,
        memetic=args.mem,
        objfunc_name=args.objfunc.upper(),
        mode=args.mode,
        img_size=args.img_size,
        display=not args.hide
    )


if __name__ == "__main__":
    main()
