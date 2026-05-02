import argparse
import logging
import time
import numpy as np
import pygame
import cv2
import os
from PIL import Image

from metaheuristic_designer.algorithms import Algorithm, MemeticAlgorithm
from metaheuristic_designer.operators import create_operator
from metaheuristic_designer.initializers import UniformInitializer, ExtendedInitializer
from metaheuristic_designer.parent_selection import create_parent_selection
from metaheuristic_designer.survivor_selection import create_survivor_selection
from metaheuristic_designer.encodings import ImageEncoding, PSOEncoding, CompositeEncoding
from metaheuristic_designer.constraint_handlers import BounceBoundConstraint, ExtendedConstraintHandler
from metaheuristic_designer.strategies import (
    HillClimb,
    LocalSearch,
    SA,
    ES,
    GA,
    DE,
    GaussianUMDA,
    GaussianPBIL,
    CrossEntropyMethod,
    RandomSearch,
    PSO,
    NoSearch
)
from metaheuristic_designer.benchmarks import (
    ImgApprox,
    ImgEntropy,
    ImgStd
)
from metaheuristic_designer.utils import check_random_state

available_objectives = ("MSE", "MAE", "SSIM", "NMI", "ENTROPY", "STD")
available_algorithms = ("hillclimb", "localsearch", "sa", "es", "ga", "de", "gaussianumda", "gaussianpbil", "crossentropy", "randomsearch", "pso")

def run_algorithm(alg_name, img_file_name, memetic, objfunc_name, mode, img_size, display, reporter, random_state):
    algorithm_params = {
        "stop_cond": "convergence or time_limit",
        "progress_metric": "time_limit",
        "time_limit": 500.0,
        "cpu_time_limit": 500.0,
        "neval": 3e6,
        "fit_target": 1e-10,
        "patience": 500,
        "verbose_timer": 0.5,
        "track_median": False,
        "track_worst": False,
        "track_complete": False,
        "track_diversity": False,
        "reporter": reporter,
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
        raise Exception(f'Objective function "{objfunc_name}" doesn\'t exist.')

    encoding = ImageEncoding(image_shape, color=True)

    search_strategy_map = {
        "hillclimb": HillClimb(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, encoding=encoding, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", F=10, N=5, random_state=random_state),
        ),
        "localsearch": LocalSearch(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, encoding=encoding, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", F=10, N=5, random_state=random_state),
            iterations=20,
        ),
        "sa": SA(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, encoding=encoding, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", F=10, N=20, random_state=random_state),
            iterations=100,
            temperature_init=1,
            alpha=0.997,
        ),
        "es": ES(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=encoding, random_state=random_state),
            mutation_op=create_operator("mutation.gaussian_mutation", F=10, N=5, random_state=random_state),
            cross_op=create_operator("crossover.uniform", random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            offspring_size=150,
        ),
        "ga": GA(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=encoding, random_state=random_state),
            mutation_op=create_operator("mutation.gaussian_mutation", F=10, N=5, random_state=random_state),
            cross_op=create_operator("crossover.uniform", random_state=random_state),
            parent_sel=create_parent_selection("best", amount=50, random_state=random_state),
            survivor_sel=create_survivor_selection("elitism", amount=20, random_state=random_state),
            mutation_prob=0.2,
            crossover_prob=0.8,
            random_state=random_state,
        ),
        "de": DE(
            de_operator_name="DE/best/1",
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=encoding, random_state=random_state),
            F=0.8,
            Cr=0.8,
        ),
        "gaussianumda": GaussianUMDA(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1000, encoding=encoding, random_state=random_state),
            parent_sel=create_parent_selection("best", amount=100),
            survivor_sel=create_survivor_selection("(m+n)"),
            scale=20,
            noise=1e-3,
            random_state=random_state,
        ),
        "gaussianpbil": GaussianPBIL(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1000, encoding=encoding, random_state=random_state),
            parent_sel=create_parent_selection("best", amount=100),
            survivor_sel=create_survivor_selection("(m+n)"),
            scale=20,
            lr=0.5,
            noise=1,
            random_state=random_state,
        ),
        "crossentropy": CrossEntropyMethod(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1000, encoding=encoding, random_state=random_state),
            random_state=random_state,
        ),
        "randomsearch": RandomSearch(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=encoding, random_state=random_state)
        ),
        "nosearch": NoSearch(
            UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=encoding, random_state=random_state)
        ),
    }

    if alg_name == "pso":
        pop_size = 100
        encoding = CompositeEncoding([PSOEncoding(objfunc.vecsize), encoding])
        base_constraint_handler = objfunc.constraint_handler

        objfunc.constraint_handler = ExtendedConstraintHandler(
            solution_handler=base_constraint_handler,
            param_handler_dict={"speed": BounceBoundConstraint(objfunc.vecsize)},
            encoding=encoding
        )
        abs_up_lim = np.maximum(np.abs(objfunc.low_lim), np.abs(objfunc.up_lim))
        initializer = ExtendedInitializer(
            solution_init=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, encoding=encoding, random_state=random_state),
            param_init_dict={"speed": UniformInitializer(objfunc.vecsize, -abs_up_lim, abs_up_lim)},
            encoding=encoding,
        )
        search_strategy = PSO(
            initializer=initializer,
            encoding=encoding,
            w=0.7,
            c1=1.5,
            c2=1.5
        )
    elif alg_name not in search_strategy_map:
        raise ValueError(f'Algorithm "{alg_name}" not recognized.')
    else:
        search_strategy = search_strategy_map[alg_name]

    if memetic:
        local_search = LocalSearch(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=search_strategy.initializer.pop_size, encoding=encoding),
            operator=create_operator("mutation.uniform_noise", min=-10, max=10, N=3),
            iterations=20,
        )
        alg = MemeticAlgorithm(
            objfunc=objfunc,
            search_strategy=search_strategy,
            local_search=local_search,
            improvement_selection=create_parent_selection("best", amount=5),
            keep_improved_solutions=True,
            **algorithm_params
        )
    else:
        alg = Algorithm(objfunc, search_strategy, **algorithm_params)

    # Manual optimisation loop with pygame display
    population = alg.initialize()
    alg.stopping_condition.restart()
    alg.stopping_condition.step(population)
    alg.reporter.log_init(alg)

    while not alg.stopping_condition.is_finished(alg.search_strategy.finish):
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit(0)
            src.fill("#000000")

        population = alg.step(population=population)

        alg.history_tracker.step(alg)
        alg.reporter.log_step(alg)
        alg.stopping_condition.step(alg.population)

        if display:
            image, _ = alg.best_solution(problem_space=True)
            render(image, display_dim, src)
            pygame.display.update()

    alg.reporter.log_end(alg)

    image, best_objective = alg.best_solution(problem_space=True)
    if display:
        render(image, display_dim, src)
        pygame.display.update()

    print(f"Best objective: {best_objective}")

    # Save image
    memetic_str = "M-" if memetic else ""
    if objfunc_name in ("MSE", "MAE", "SSIM", "NMI"):
        img_name = img_file_name.split("/")[-1].split(".")[0]
        out_img_name = f"{img_name}_{objfunc_name}_{image_shape[0]}x{image_shape[1]}_{memetic_str}{alg_name}.png"
    else:
        out_img_name = f"{objfunc_name}_{image_shape[0]}x{image_shape[1]}_{memetic_str}{alg_name}.png"

    if not os.path.exists("./examples/results/"):
        os.makedirs("./examples/results/")
    Image.fromarray(image.astype(np.uint8)).save("./examples/results/" + out_img_name)


def render(image, display_dim, src):
    texture = cv2.resize(image.transpose([1, 0, 2]), display_dim, interpolation=cv2.INTER_NEAREST)
    pygame.surfarray.blit_array(src, texture)
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest="algorithm", help=f"Specify an algorithm. Available options are {available_algorithms}.", default="ga")
    parser.add_argument(
        "-m",
        "--memetic",
        dest="mem",
        action="store_true",
        help="Does local search after mutation",
    )
    parser.add_argument("-i", "--image", dest="img", default="data/images/cat.png", help="Path to reference image.")
    parser.add_argument("-s", "--img_size", default="32,32", help="Image size as 'H,W'.")
    parser.add_argument("--hide", action="store_true", help="Disable real‑time display.")
    parser.add_argument("--mode", default="min", help="'min' or 'max'.")
    parser.add_argument("-o", "--objective", dest="objective", help=f"Name of the objective function. Available options are {available_objectives}", default="MSE")
    parser.add_argument("-r", "--seed", dest="seed", help="Random seed to use", default=42, type=int)
    parser.add_argument("--log", default="WARNING", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("-v", "--reporter", default="tqdm", help="Reporter to use for progress tracking.")
    args = parser.parse_args()

    rng = check_random_state(args.seed)
    logging.basicConfig()
    logging.getLogger("metaheuristic_designer").setLevel(args.log.upper())

    run_algorithm(
        alg_name=args.algorithm.lower(),
        img_file_name=args.img,
        memetic=args.mem,
        objfunc_name=args.objective.upper(),
        mode=args.mode,
        img_size=args.img_size,
        display=not args.hide,
        reporter=args.reporter,
        random_state=rng,
    )


if __name__ == "__main__":
    main()