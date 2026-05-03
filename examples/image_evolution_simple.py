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


def run_algorithm(alg_name, img_path, img_size, obj_name, ngen, display, seed):
    rng = mhd.check_random_state(seed)
    img_shape = (img_size, img_size)

    # ---- objective function ----
    if obj_name in ("MSE", "MAE"):
        ref_img = Image.open(img_path)
        objfunc = ImgApprox(img_shape, ref_img, diff_func=obj_name, mode="min")
    elif obj_name == "ENTROPY":
        objfunc = ImgEntropy(img_shape, 256, mode="max")
    elif obj_name == "STD":
        objfunc = ImgStd(img_shape, mode="max")
    else:
        raise ValueError(f"Unknown objective: {obj_name}")

    # Encoding for images
    img_enc = ImageEncoding(img_shape, color=True)

    # Shared parameters for the simple wrapper
    algo_params = {
        "stop_cond": "ngen",
        "ngen": ngen,
        "reporter": "tqdm",
        "random_state": rng,
        "encoding": img_enc,
    }

    # Build the algorithm eagerly – same pattern as exec_basic.py
    alg_map = {
        "HillClimb": simple.hill_climb_real(objfunc, **algo_params),
        "SA":        simple.simulated_annealing_real(objfunc, **algo_params),
        "ES":        simple.evolution_strategy_real(objfunc, **algo_params),
        "GA":        simple.genetic_algorithm_real(objfunc, **algo_params),
        "DE":        simple.differential_evolution_real(objfunc, **algo_params),
        "PSO":       simple.particle_swarm_real(objfunc, **algo_params),
        "RandomSearch": simple.random_search_real(objfunc, **algo_params),
    }
    if alg_name not in alg_map:
        raise ValueError(f"Unknown algorithm: {alg_name}")
    algo = alg_map[alg_name]

    # ---- Pygame display setup (same as original image_evolution.py) ----
    display_dim = [600, 600]
    if display:
        pygame.init()
        src = pygame.display.set_mode(display_dim)
        pygame.display.set_caption("Evo graphics")

    # ---- Manual optimisation loop with display ----
    population = algo.initialize()
    algo.stopping_condition.restart()
    algo.stopping_condition.step(population)
    algo.reporter.log_init(algo)

    while not algo.stopping_condition.is_finished(algo.search_strategy.finish):
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit(0)
            src.fill("#000000")

        population = algo.step(population=population)

        algo.history_tracker.step(algo)
        algo.reporter.log_step(algo)
        algo.stopping_condition.step(algo.population)

        if display:
            image, _ = algo.best_solution(problem_space=True)
            _render(image, display_dim, src)
            pygame.display.update()

    algo.reporter.log_end(algo)

    # Final best image
    best_img, best_objective = algo.best_solution(problem_space=True)
    if display:
        _render(best_img, display_dim, src)
        pygame.display.update()

    print(f"Best {obj_name}: {best_objective:.2f}")

    # Save result image (same naming as original)
    if not os.path.exists("./examples/results/"):
        os.makedirs("./examples/results/")
    img_name = img_path.split("/")[-1].split(".")[0]
    out_name = f"{img_name}_{obj_name}_{img_shape[0]}x{img_shape[1]}_{alg_name}.png"
    Image.fromarray(best_img.astype(np.uint8)).save(f"./examples/results/{out_name}")
    print(f"Result saved to ./examples/results/{out_name}")


def _render(image, display_dim, src):
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
    parser.add_argument("-a", "--algorithm", default="GA",
                        choices=["HillClimb","SA","ES","GA","DE","PSO","RandomSearch"])
    parser.add_argument("-i", "--image", default="data/images/cat.png")
    parser.add_argument("-s", "--img-size", type=int, default=32,
                        help="Width and height of the image")
    parser.add_argument("-o", "--objective", default="MSE",
                        choices=["MSE","MAE","ENTROPY","STD"])
    parser.add_argument("--ngen", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hide", action="store_true", help="Disable real‑time display")
    args = parser.parse_args()

    run_algorithm(
        alg_name=args.algorithm,
        img_path=args.image,
        img_size=args.img_size,
        obj_name=args.objective,
        ngen=args.ngen,
        display=not args.hide,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()