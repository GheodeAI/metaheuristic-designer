import argparse
import logging
import time
import numpy as np
import pygame
import cv2
import os
from PIL import Image

import metaheuristic_designer as mhd
from metaheuristic_designer.benchmarks import ImgApprox, ImgEntropy, ImgStd
from metaheuristic_designer.encodings import ImageEncoding
from metaheuristic_designer import simple
from metaheuristic_designer.utils import check_random_state


def run_algorithm(alg_name, img_file_name, img_size, obj_name, mode, ngen, display, seed):
    rng = mhd.check_random_state(seed)
    img_shape = (img_size, img_size)

    # ---- objective function ----
    if obj_name in ("mse", "mae"):
        reference_img = Image.open(img_file_name)
        img_name = img_file_name.split("/")[-1]
        img_name = img_name.split(".")[0]
        objfunc = ImgApprox(img_shape, reference_img, img_name=img_name, diff_func=obj_name)
    elif obj_name == "entropy":
        objfunc = ImgEntropy(img_shape, 256, mode="max")
    elif obj_name == "std":
        objfunc = ImgStd(img_shape, mode="max")
    else:
        raise ValueError(f"Unknown objective: {obj_name}")
    print(objfunc.vecsize)

    # Encoding for images
    img_enc = ImageEncoding(img_shape, color=True)

    # Shared parameters for the simple wrapper
    algo_params = {
        # "stop_cond": "convergence or time_limit",
        "stop_cond": "time_limit",
        "progress_metric": "time_limit",
        "time_limit": 500.0,
        "patience": 500,
        "reporter": "tqdm",
        "random_state": rng,
        "encoding": img_enc,
    }

    # Build the algorithm eagerly – same pattern as exec_basic.py
    alg_map = {
        "hillclimb": simple.hill_climb_discrete(objfunc, **algo_params),
        "localsearch": simple.local_search_real(objfunc, **algo_params),
        "sa":        simple.simulated_annealing_real(objfunc, **algo_params),
        "es":        simple.evolution_strategy_real(objfunc, **algo_params),
        "ga":        simple.genetic_algorithm_real(objfunc, **algo_params),
        "de":        simple.differential_evolution_real(objfunc, **algo_params),
        "pso":       simple.particle_swarm_real(objfunc, **algo_params),
        "randomsearch": simple.random_search_real(objfunc, **algo_params),
    }
    if alg_name not in alg_map:
        raise ValueError(f"Unknown algorithm: {alg_name}")
    alg = alg_map[alg_name]

    # ---- Pygame display setup ----
    display_dim = [600, 600]
    if display:
        pygame.init()
        src = pygame.display.set_mode(display_dim)
        pygame.display.set_caption("Evo graphics")

    # ---- Manual optimization loop with display ----
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

    # Final best image
    best_img, best_objective = alg.best_solution(problem_space=True)
    if display:
        render(best_img, display_dim, src)
        pygame.display.update()

    print(f"Best {obj_name}: {best_objective:.2f}")

    # Save result image (same naming as original)
    if not os.path.exists("./examples/results/"):
        os.makedirs("./examples/results/")
    img_name = img_file_name.split("/")[-1].split(".")[0]
    out_name = f"{img_name}_{obj_name}_{img_shape[0]}x{img_shape[1]}_{alg_name}.png"
    Image.fromarray(best_img.astype(np.uint8)).save(f"./examples/results/{out_name}")
    print(f"Result saved to ./examples/results/{out_name}")


def render(image, display_dim, src):
    """Helper to draw an image onto the pygame surface."""
    texture = cv2.resize(
        image.transpose([1, 0, 2]),
        display_dim,
        interpolation=cv2.INTER_NEAREST,
    )
    pygame.surfarray.blit_array(src, texture)
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="ga", type=str.lower,
                        choices=["hillclimb","sa","es","ga","de","pso","randomsearch","localsearch"])
    parser.add_argument("-i", "--image", default="data/images/cat.png")
    parser.add_argument("-s", "--img-size", type=int, default=32,
                        help="Width and height of the image")
    parser.add_argument("--mode", default="min", help="'min' or 'max'.")
    parser.add_argument("-o", "--objective", default="mse", type=str.lower,
                        choices=["mse","mae","entropy","std"])
    parser.add_argument("--ngen", type=int, default=1000)
    parser.add_argument("-r", "--seed", dest="seed", help="Random seed to use", default=None, type=int)
    parser.add_argument("--log", default="WARNING", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--hide", action="store_true", help="Disable real-time display")
    args = parser.parse_args()

    rng = check_random_state(args.seed)
    logging.basicConfig()
    logging.getLogger("metaheuristic_designer").setLevel(args.log.upper())

    run_algorithm(
        alg_name=args.algorithm.lower(),
        img_file_name=args.image,
        img_size=args.img_size,
        obj_name=args.objective.lower(),
        mode=args.mode,
        ngen=args.ngen,
        display=not args.hide,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()