from metaheuristic_designer import ObjectiveFunc, ParamScheduler
from metaheuristic_designer.algorithms import GeneralAlgorithm, MemeticAlgorithm
from metaheuristic_designer.operators import OperatorVector
from metaheuristic_designer.operators import OperatorVector
from metaheuristic_designer.strategies import *
from metaheuristic_designer.initializers import *
from metaheuristic_designer.selectionMethods import ParentSelection, SurvivorSelection, ParentSelectionNull
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
        "stop_cond": "convergence or time_limit",
        "progress_metric": "time_limit",
        "time_limit": 500.0,
        "patience": 500,
        "verbose": True,
        "v_timer": 0.5,

        # Parallel
        "parallel": False,
        "threads": 8,
    }

    # Window size
    display_dim = [600, 600]

    # Image size
    image_shape = tuple(map(int, img_size.split(",")))

    # Initialize display
    if display:
        pygame.init()
        src = pygame.display.set_mode(display_dim)
        pygame.display.set_caption("Evo graphics")

    # Initialize objective function
    match objfunc_name:
        case "MSE" | "MAE" | "SSIM" | "NMI":
            # Load reference image
            reference_img = Image.open(img_file_name)
            img_name = img_file_name.split("/")[-1]
            img_name = img_name.split(".")[0]

            objfunc = ImgApprox(image_shape, reference_img, img_name=img_name, diff_func=objfunc_name, mode=mode)
        case "ENTROPY":
            objfunc = ImgEntropy(image_shape, 256, mode=mode)
        case "STD":
            objfunc = ImgStd(image_shape, mode=mode)

    # Prepare algorithm
    encoding = ImageEncoding(image_shape, color=True)

    match alg_name:
        case "HILLCLIMB":
            search_strat = HillClimb(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, encoding=encoding),
                operator=OperatorVector("MutNoise", {"distrib": "Normal", "F": 10, "N": 5}),
            )
        case "LOCALSEARCH":
            search_strat = LocalSearch(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, encoding=encoding),
                operator=OperatorVector("MutNoise", {"distrib": "Normal", "F": 10, "N": 5}),
                params={"iters": 20},
            )
        case "SA":
            search_strat = SA(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, encoding=encoding),
                operator=OperatorVector("MutNoise", {"distrib": "Normal", "F": 10, "N": 5}),
                params={"iter": 100, "temp_init": 1, "alpha": 0.997},
            )
        case "ES":
            pop_size = 100
            lam = 150
            search_strat = ES(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                mutation_op=OperatorVector("MutNoise", {"distrib": "Normal", "F": 10, "N": 5}),
                cross_op=OperatorVector("Multipoint"),
                survivor_sel=SurvivorSelection("(m+n)"),
                params={"offspringSize": lam},
            )
        case "GA":
            pop_size = 100
            n_parents = 50
            n_elites = 20
            search_strat = GA(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                mutation_op=OperatorVector("MutNoise", {"distrib": "Normal", "F": 10, "N": 5}),
                cross_op=OperatorVector("Multipoint"),
                parent_sel=ParentSelection("Best", {"amount": n_parents}),
                survivor_sel=SurvivorSelection("Elitism", {"amount": n_elites}),
                params={"pcross": 0.8, "pmut": 0.2},
            )
        case "HS":
            pop_size = 100
            params["patience"] = 1000
            search_strat = HS(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                params={"HMCR": 0.8, "BW": 0.5, "PAR": 0.2},
            )
        case "DE":
            pop_size = 100
            search_strat = DE(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                de_operator=OperatorVector("DE/best/1", {"F": 0.8, "Cr": 0.8}),
            )
        case "PSO":
            pop_size = 100
            search_strat = PSO(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                params={"w": 0.7, "c1": 1.5, "c2": 1.5},
            )
        case "BINOMIALUMDA":
            search_strat = BinomialUMDA(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                parent_sel=ParentSelection("Best", {"amount": n_parents}),
                survivor_sel=SurvivorSelection("(m+n)"),
                params={"n": 256, "noise": 0},
            )
        case "BINOMIALPBIL":
            search_strat = BinomialPBIL(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                parent_sel=ParentSelection("Best", {"amount": n_parents}),
                survivor_sel=SurvivorSelection("(m+n)"),
                params={"n": 256, "lr": 0.5, "noise": 0},
            )
        case "GAUSSIANUMDA":
            pop_size = 100
            n_parents = 20
            search_strat = GaussianUMDA(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                parent_sel=ParentSelection("Best", {"amount": n_parents}),
                survivor_sel=SurvivorSelection("(m+n)"),
                params={"scale": 0.1, "noise": 1e-3},
            )
        case "GAUSSIANPBIL":
            pop_size = 100
            n_parents = 20
            search_strat = GaussianPBIL(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                parent_sel=ParentSelection("Best", {"amount": n_parents}),
                survivor_sel=SurvivorSelection("(m+n)"),
                params={"scale": 0.1, "lr": 0.3, "noise": 1e-3},
            )
        case "CROSSENTROPY":
            pop_size = 1000
            search_strat = CrossEntropyMethod(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
            )
        case "CRO":
            pop_size = 100
            search_strat = CRO(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                mutation_op=OperatorVector("MutNoise", {"distrib": "Cauchy", "F": 1e-3, "N": 1}),
                cross_op=OperatorVector("Multipoint"),
                params={"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
            )
        case "CRO_SL":
            pop_size = 100
            DEparams = {"F": 0.7, "Cr": 0.8}
            search_strat = CRO_SL(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                operator_list=[
                    OperatorVector("DE/rand/1", DEparams),
                    OperatorVector("DE/best/2", DEparams),
                    OperatorVector("DE/current-to-best/1", DEparams),
                    OperatorVector("DE/current-to-rand/1", DEparams),
                ],
                params={"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
            )
        case "PCRO_SL":
            pop_size = 100
            DEparams = {"F": 0.7, "Cr": 0.8}
            search_strat = PCRO_SL(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                operator_list=[
                    OperatorVector("DE/rand/1", DEparams),
                    OperatorVector("DE/best/2", DEparams),
                    OperatorVector("DE/current-to-best/1", DEparams),
                    OperatorVector("DE/current-to-rand/1", DEparams),
                ],
                params={"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
            )
        case "DPCRO_SL":
            pop_size = 100
            DEparams = {"F": 0.7, "Cr": 0.8}
            search_strat_params = {
                "rho": 0.6,
                "Fb": 0.95,
                "Fd": 0.1,
                "Pd": 0.9,
                "attempts": 3,
                "group_subs": True,
                "dyn_method": "diff",
                "dyn_metric": "best",
                "dyn_steps": 75,
                "prob_amp": 0.1,
            }
            search_strat = DPCRO_SL(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                operator_list=[
                    OperatorVector("DE/rand/1", DEparams),
                    OperatorVector("DE/best/2", DEparams),
                    OperatorVector("DE/current-to-best/1", DEparams),
                    OperatorVector("DE/current-to-rand/1", DEparams),
                ],
                params=search_strat_params,
            )
        case "RVNS":
            pop_size = 1
            neighborhood_structures = [
                OperatorVector(
                    "MutNoise",
                    {"distrib": "Uniform", "min": -10, "max": 10, "N": n},
                    name=f"UniformSample(N={n:0.0f})",
                )
                for n in np.logspace(5, 12, base=2, num=200)
            ]
            search_strat = RVNS(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                op_list=neighborhood_structures,
            )
        case "VND":
            pop_size = 1
            neighborhood_structures = [
                OperatorVector(
                    "MutNoise",
                    {"distrib": "Uniform", "min": -10, "max": 10, "N": n},
                    name=f"UniformSample(N={n:0.0f})",
                )
                for n in np.logspace(5, 12, base=2, num=200)
            ]
            search_strat = VND(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                op_list=neighborhood_structures,
            )
        case "VNS":
            pop_size = 1
            neighborhood_structures = [
                OperatorVector(
                    "MutNoise",
                    {"distrib": "Uniform", "min": -10, "max": 10, "N": n},
                    name=f"UniformSample(N={n:0.0f})",
                )
                for n in np.logspace(5, 12, base=2, num=200)
            ]
            local_search = LocalSearch(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                params={"iters": 20},
            )
            search_strat = VNS(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                op_list=neighborhood_structures,
                local_search=local_search,
                params={"nchange": "seq"},
                inner_loop_params={
                    "stop_cond": "convergence",
                    "patience": 3,
                    "verbose": params["verbose"],
                    "v_timer": params["v_timer"],
                },
            )
        case "GVNS":
            pop_size = 1
            neighborhood_structures = [
                OperatorVector(
                    "MutNoise",
                    {"distrib": "Uniform", "min": -10, "max": 10, "N": n},
                    name=f"UniformSample(N={n:0.0f})",
                )
                for n in np.logspace(5, 12, base=2, num=200)
            ]
            local_search = VND(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                op_list=neighborhood_structures,
            )
            search_strat = VNS(
                initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
                op_list=neighborhood_structures,
                local_search=local_search,
                params={"nchange": "pipe"},
                inner_loop_params={
                    "stop_cond": "convergence",
                    "patience": 500,
                    "verbose": params["verbose"],
                    "v_timer": params["v_timer"],
                },
            )
        case "RANDOMSEARCH":
            pop_size = 100
            search_strat = RandomSearch(
                UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding)
            )
        case "NOSEARCH":
            pop_size = 100
            search_strat = NoSearch(UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding))
        case _:
            raise Exception(f'Algorithm "{alg_name}" doesn\'t exist.')

    if memetic:
        local_search = LocalSearch(
            initializer=UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding),
            operator=OperatorVector("MutRand", {"distrib": "Uniform", "min": -10, "max": -10, "N": 3}),
            params={"iters": 20},
        )
        alg = MemeticAlgorithm(objfunc, search_strat, local_search, mem_select, params=params)
    else:
        alg = GeneralAlgorithm(objfunc, search_strat, params=params)

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
        pygame.display.update()
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
    parser.add_argument(
        "-o", "--objfunc", dest="objfunc", help="Which objective function to use ('MSE', 'MAE', 'SSIM', 'NMI', 'STD', 'Entropy').", default="mse"
    )
    args = parser.parse_args()

    print(args.hide)

    run_algorithm(
        alg_name=args.algorithm.upper(),
        img_file_name=args.img,
        memetic=args.mem,
        objfunc_name=args.objfunc.upper(),
        mode=args.mode,
        img_size=args.img_size,
        display=not args.hide,
    )


if __name__ == "__main__":
    main()
