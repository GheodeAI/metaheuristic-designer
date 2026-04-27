import argparse
import logging
import time
import numpy as np
import pygame
import cv2
import os
from PIL import Image

from metaheuristic_designer.algorithms import StandardAlgorithm, MemeticAlgorithm
from metaheuristic_designer.operators import create_operator
from metaheuristic_designer.initializers import *
from metaheuristic_designer.parent_selection_methods import create_parent_selection
from metaheuristic_designer.survivor_selection_methods import create_survivor_selection
from metaheuristic_designer.encodings import ImageEncoding
from metaheuristic_designer.strategies import *
from metaheuristic_designer.benchmarks import *
from metaheuristic_designer.utils import check_random_state


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
    # Algorithm parameters
    algorithm_params = {
        "stop_cond": "convergence or time_limit",
        "progress_metric": "time_limit",
        "time_limit": 500.0,
        "patience": 500,
        "verbose": True,
        "v_timer": 0.5,
        "parallel": False,
        "threads": 8,
    }

    display_dim = [600, 600]
    image_shape = tuple(map(int, img_size.split(",")))

    if display:
        pygame.init()
        src = pygame.display.set_mode(display_dim)
        pygame.display.set_caption("Evo graphics")

    if objfunc_name in ("MSE", "MAE", "SSIM", "NMI"):
        reference_img = Image.open(img_file_name)
        img_name = img_file_name.split("/")[-1]
        img_name = img_name.split(".")[0]
        objfunc = ImgApprox(image_shape, reference_img, img_name=img_name, diff_func=objfunc_name, mode=mode)
    elif objfunc_name == "ENTROPY":
        objfunc = ImgEntropy(image_shape, 256, mode=mode)
    elif objfunc_name == "STD":
        objfunc = ImgStd(image_shape, mode=mode)
    else:
        raise ValueError(f'Objective function "{objfunc_name}" not available.')

    encoding = ImageEncoding(image_shape, color=True)

    search_strategy_map = {
        "HILLCLIMB": HillClimb(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, encoding=encoding),
            operator=create_operator("mutation.gaussian_mutation", F=10, N=5),
        ),
        "LOCALSEARCH": LocalSearch(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, encoding=encoding),
            operator=create_operator("mutation.gaussian_mutation", F=10, N=5),
            iterations=20,
        ),
        "SA": SA(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, encoding=encoding),
            operator=create_operator("mutation.gaussian_mutation", F=10, N=5),
            iterations=100,
            temperature_init=1,
            alpha=0.997,
        ),
        "ES": ES(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=encoding),
            mutation_op=create_operator("mutation.gaussian_mutation", F=10, N=5),
            cross_op=create_operator("crossover.uniform"),
            survivor_sel=create_survivor_selection("(m+n)"),
            offspring_size=150,
        ),
        "GA": GA(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=encoding),
            mutation_op=create_operator("mutation.gaussian_mutation", F=10, N=5),
            cross_op=create_operator("crossover.uniform"),
            parent_sel=create_parent_selection("best", amount=50),
            survivor_sel=create_survivor_selection("elitism", amount=20),
            mutation_prob=0.2,
            crossover_prob=0.8,
        ),
        "DE": DE(
            de_operator_name="DE/best/1",
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=encoding),
            F=0.8,
            Cr=0.8,
        ),
        "BINOMIALUMDA": BinomialUMDA(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1000, encoding=encoding),
            parent_sel=create_parent_selection("best", amount=100),
            survivor_sel=create_survivor_selection("(m+n)"),
            n=256,
            noise=0,
        ),
        "BINOMIALPBIL": BinomialPBIL(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1000, encoding=encoding),
            parent_sel=create_parent_selection("best", amount=100),
            survivor_sel=create_survivor_selection("(m+n)"),
            n=256,
            lr=0.5,
            noise=1e-4,
        ),
        "GAUSSIANUMDA": GaussianUMDA(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1000, encoding=encoding),
            parent_sel=create_parent_selection("best", amount=100),
            survivor_sel=create_survivor_selection("(m+n)"),
            scale=20,
            noise=1e-3,
        ),
        "GAUSSIANPBIL": GaussianPBIL(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1000, encoding=encoding),
            parent_sel=create_parent_selection("best", amount=100),
            survivor_sel=create_survivor_selection("(m+n)"),
            scale=20,
            lr=0.5,
            noise=1,
        ),
        "CROSSENTROPY": CrossEntropyMethod(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1000, encoding=encoding),
        ),
        "RANDOMSEARCH": RandomSearch(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=encoding)
        ),
        "NOSEARCH": NoSearch(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=encoding)
        ),
    }

    if alg_name not in search_strategy_map:
        raise ValueError(f'Algorithm "{alg_name}" not recognized.')
    search_strategy = search_strategy_map[alg_name]

    # ---- Memetic branch ----
    if memetic:
        mem_select = create_parent_selection("best", amount=5)
        local_search = LocalSearch(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=search_strategy.initializer.pop_size, encoding=encoding),
            operator=create_operator("mutation.uniform_noise", min=-10, max=10, N=3),
            iterations=20,
        )
        alg = MemeticAlgorithm(objfunc, search_strategy, local_search, mem_select, **algorithm_params)
    else:
        alg = StandardAlgorithm(objfunc, search_strategy, **algorithm_params)

    # ---- Optimisation loop with real‑time display ----
    real_time_start = time.time()
    cpu_time_start = time.process_time()
    display_timer = time.time()

    population = alg.initialize()
    alg.update(skip_step=False)

    while not alg.ended:
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit(0)
            src.fill("#000000")

        population = alg.step(population=population)
        alg.update()

        if alg.verbose and time.time() - display_timer > alg.v_timer:
            alg.step_info(real_time_start)
            display_timer = time.time()

        if display:
            image, _ = population.best_solution(decoded=True)
            render(image, display_dim, src)
            pygame.display.update()

    alg.real_time_spent = time.time() - real_time_start
    alg.cpu_time_spent = time.process_time() - cpu_time_start

    image, _ = alg.best_solution(decoded=True)
    if display:
        render(image, display_dim, src)
        pygame.display.update()
    alg.display_report(show_plots=True)

    # ---- Save result ----
    memetic_str = "M-" if memetic else ""
    if objfunc_name in ("MSE", "MAE", "SSIM", "NMI"):
        img_name = img_file_name.split("/")[-1].split(".")[0]
        out_img_name = f"{img_name}_{objfunc_name}_{image_shape[0]}x{image_shape[1]}_{memetic_str}{alg_name}.png"
    else:
        out_img_name = f"{objfunc_name}_{image_shape[0]}x{image_shape[1]}_{memetic_str}{alg_name}.png"

    save_to_image(image, out_img_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="SA", help="Algorithm name.")
    parser.add_argument("-i", "--image", dest="img", default="data/images/cat.png", help="Path to reference image.")
    parser.add_argument("-s", "--img_size", default="32,32", help="Image size as 'H,W'.")
    parser.add_argument("-m", "--memetic", action="store_true", help="Apply memetic wrapper.")
    parser.add_argument("--hide", action="store_true", help="Disable real‑time display.")
    parser.add_argument("--mode", default="min", help="'min' or 'max'.")
    parser.add_argument("-o", "--objfunc", choices=["MSE","MAE","SSIM","NMI","STD","ENTROPY"], default="MSE",
                        help="Objective function.")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log", default="WARNING", help="Log level.")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig()
    logging.getLogger("metaheuristic_designer").setLevel(args.log.upper())

    run_algorithm(
        alg_name=args.algorithm.upper(),
        img_file_name=args.img,
        memetic=args.memetic,
        objfunc_name=args.objfunc.upper(),
        mode=args.mode,
        img_size=args.img_size,
        display=not args.hide,
    )


if __name__ == "__main__":
    main()