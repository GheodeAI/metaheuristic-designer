import argparse
import logging
import numpy as np
import pygame
import cv2
import os
from PIL import Image

from metaheuristic_designer.algorithms import Algorithm
from metaheuristic_designer.operators import create_operator
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.parent_selection import create_parent_selection
from metaheuristic_designer.survivor_selection import create_survivor_selection
from metaheuristic_designer.encodings import ImageEncoding, PSOEncoding
from metaheuristic_designer.strategies import (
    HillClimb,
    LocalSearch,
    SA,
    ES,
    GA,
    DE,
    CMA_ES,
    GaussianUMDA,
    GaussianPBIL,
    CrossEntropyMethod,
    RandomSearch,
    PSO,
    NoSearch,
    MemeticStrategy
)
from metaheuristic_designer.benchmarks import ImgApprox, ImgEntropy, ImgStd
from metaheuristic_designer.utils import check_rng

available_objectives = ("MSE", "MAE", "SSIM", "NMI", "ENTROPY", "STD")
available_algorithms = ("hillclimb", "localsearch", "sa", "es", "ga", "de", "gaussianumda", "gaussianpbil", "crossentropy", "randomsearch", "pso")


def run_algorithm(alg_name, img_file_name, memetic, objfunc_name, mode, img_size, display, reporter, evaluations, rng):
    image_shape = tuple(map(int, img_size.split(",")))

    if evaluations is None:
        evaluations = image_shape[0] * image_shape[1] * 2000

    algorithm_params = {
        "stop_condition_str": "convergence or max_evaluations",
        "progress_metric_str": "max_evaluations",
        "max_evaluations": evaluations,
        "max_patience": 100000,
        "reporter": reporter
    }

    if mode == "max" and objfunc_name in ("MSE", "MAE"):
        print(f"Maximizing {objfunc_name} might not yield meaningful results.")
    elif mode == "min" and objfunc_name in ("NMI", "SSIM"):
        print(f"Minimizing {objfunc_name} might not yield meaningful results.")


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
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, encoding=encoding, rng=rng
            ),
            operator=create_operator("mutation.gaussian_mutation", F=10, N=5, rng=rng),
        ),
        "localsearch": LocalSearch(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, encoding=encoding, rng=rng
            ),
            operator=create_operator("mutation.gaussian_mutation", F=10, N=5, rng=rng),
            iterations=20,
        ),
        "sa": SA(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, encoding=encoding, rng=rng
            ),
            operator=create_operator("mutation.gaussian_mutation", F=10, N=20, rng=rng),
            iterations=250,
            temperature_init=1,
            alpha=0.997,
        ),
        "es": ES(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, encoding=encoding, rng=rng
            ),
            mutation_op=create_operator("mutation.gaussian_mutation", F=10, N=5, rng=rng),
            crossover_op=create_operator("crossover.uniform", rng=rng),
            survivor_sel=create_survivor_selection("(m+n)", rng=rng),
            offspring_size=150,
        ),
        "ga": GA(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, encoding=encoding, rng=rng
            ),
            mutation_op=create_operator("mutation.gaussian_mutation", F=10, N=5, rng=rng),
            crossover_op=create_operator("crossover.uniform", rng=rng),
            parent_sel=create_parent_selection("best", amount=50, rng=rng),
            survivor_sel=create_survivor_selection("elitism", amount=20, rng=rng),
            mutation_prob=0.2,
            crossover_prob=0.8,
            rng=rng,
        ),
        "de": DE(
            de_operator_name="DE/best/1",
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, encoding=encoding, rng=rng
            ),
            F=0.8,
            Cr=0.8,
        ),
        "cmaes": CMA_ES(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, encoding=encoding, rng=rng),
            rng=rng,
        ),
        "gaussianumda": GaussianUMDA(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1000, encoding=encoding, rng=rng
            ),
            parent_sel=create_parent_selection("best", amount=100),
            survivor_sel=create_survivor_selection("(m+n)"),
            scale=20,
            noise=1e-3,
            rng=rng,
        ),
        "gaussianpbil": GaussianPBIL(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1000, encoding=encoding, rng=rng
            ),
            parent_sel=create_parent_selection("best", amount=100),
            survivor_sel=create_survivor_selection("(m+n)"),
            scale=20,
            lr=0.5,
            noise=1,
            rng=rng,
        ),
        "crossentropy": CrossEntropyMethod(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1000, encoding=encoding, rng=rng
            ),
            rng=rng,
        ),
        "randomsearch": RandomSearch(
            UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, encoding=encoding, rng=rng
            )
        ),
        "nosearch": NoSearch(
            UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, encoding=encoding, rng=rng
            )
        ),
    }

    if alg_name == "pso":
        search_strategy = PSO(
            initializer=UniformInitializer(
                objfunc.dimension,
                objfunc.lower_bound,
                objfunc.upper_bound,
                population_size=600,
                encoding=PSOEncoding(objfunc.dimension),
                rng=rng,
            ),
            encoding=encoding,
            w=0.5,
            c1=1.5,
            c2=1.5,
        )
    elif alg_name not in search_strategy_map:
        raise ValueError(f'Algorithm "{alg_name}" not recognized.')
    else:
        search_strategy = search_strategy_map[alg_name]

    if memetic:
        local_search = LocalSearch(
            initializer=UniformInitializer(
                objfunc.dimension,
                objfunc.lower_bound,
                objfunc.upper_bound,
                population_size=search_strategy.initializer.population_size,
                encoding=encoding,
            ),
            operator=create_operator("mutation.uniform_noise", min=-10, max=10),
            iterations=20,
        )
        search_strategy = MemeticStrategy(
            main_strategy=search_strategy,
            local_search_heuristic=local_search,
            local_search_depth=10,
            local_search_frequency=5,
            improvement_selection=create_parent_selection("best", amount=5),
            keep_improved_solutions=True,
            rng=rng,
        )
    alg = Algorithm(objfunc, search_strategy, **algorithm_params)

    # Pygame display setup
    display_dim = [600, 600]
    if display:
        pygame.init()
        src = pygame.display.set_mode(display_dim)
        pygame.display.set_caption("Evo graphics")

    # Manual optimization loop with pygame display
    alg.restart()
    population = alg.initialize()

    while not alg.stopping_condition.is_finished(alg.search_strategy.finish):
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit(0)
            src.fill("#000000")

        population = alg.step(prev_population=population)

        if display:
            image, _ = alg.best_solution()
            render(image, display_dim, src)
            pygame.display.update()

    alg.reporter.log_end(alg)

    image, best_objective = alg.best_solution()
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
    parser.add_argument(
        "-a", "--algorithm", dest="algorithm", help=f"Specify an algorithm. Available options are {available_algorithms}.", default="es"
    )
    parser.add_argument(
        "-m",
        "--memetic",
        dest="mem",
        action="store_true",
        help="Does local search after mutation",
    )
    parser.add_argument("-i", "--image", dest="img", default="data/images/cat.png", help="Path to reference image.")
    parser.add_argument("-s", "--img_size", default="24,24", help="Image size as 'H,W'.")
    parser.add_argument("--hide", action="store_true", help="Disable real-time display.")
    parser.add_argument("--mode", default=None, help="'min' or 'max'.")
    parser.add_argument(
        "-o", "--objective", dest="objective", help=f"Name of the objective function. Available options are {available_objectives}", default="SSIM"
    )
    parser.add_argument("-e", "--evaluations", default=None, help="Maximum number of evaluations.", type=int)
    parser.add_argument("-r", "--seed", dest="seed", help="Random seed to use", default=42, type=int)
    parser.add_argument("--log", default="WARNING", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("-v", "--reporter", default="tqdm", help="Reporter to use for progress tracking.")
    args = parser.parse_args()

    rng = check_rng(args.seed)
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
        evaluations=args.evaluations,
        rng=rng,
    )


if __name__ == "__main__":
    main()
