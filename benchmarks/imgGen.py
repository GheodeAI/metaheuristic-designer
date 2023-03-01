import sys
sys.path.append("../..")

from PyMetaheuristics import GeneralSearch, ObjectiveFunc, ParentSelection, SurvivorSelection, ParamScheduler
from PyMetaheuristics.Operators import OperatorReal, OperatorInt, OperatorBinary
from PyMetaheuristics.Algorithms import *
from imgProblem import * 

import pygame
import time
import numpy as np
import cv2
import os
from copy import deepcopy
from PIL import Image

import matplotlib
matplotlib.use("TkAgg")

import argparse


def render(image, display_dim, src):
    texture = cv2.resize(image.astype(np.uint8).transpose([1,0,2]), display_dim, interpolation = cv2.INTER_NEAREST)
    pygame.surfarray.blit_array(src, texture)
    pygame.display.flip()

def save_to_image(image, img_name="result.png"):
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    filename = './results/' + img_name
    Image.fromarray(image.astype(np.uint8)).save(filename)

def run_algorithm(alg_name, img_file_name):
    params = {
        # General
        "stop_cond": "neval",
        "time_limit": 20.0,
        "ngen": 1000,
        "neval": 3e5,
        "fit_target": 0,

        "verbose": True,
        "v_timer": 0.5
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

    mutation_op = OperatorInt("MutRand", {"method": "Uniform", "Low":-10, "Up":10, "N":500})
    cross_op = OperatorInt("Multicross", {"N": 3})
    parent_sel_op = ParentSelection("Best", {"amount": 20})
    selection_op = SurvivorSelection("(m+n)")

    if alg_name == "HillClimb":
        search_strat = HillClimb(objfunc, mutation_op)
    elif alg_name == "LocalSearch":
        search_strat = LocalSearch(objfunc, mutation_op, {"iters":20})
    elif alg_name == "ES":
        search_strat = ES(objfunc, mutation_op, cross_op, parent_sel_op, selection_op, {"popSize":100, "offspringSize":500})
    elif alg_name == "GA":
        search_strat = GA(objfunc, mutation_op, cross_op, parent_sel_op, selection_op, {"popSize":100, "pcross":0.8, "pmut":0.2})
    elif alg_name == "SA":
        search_strat = SA(objfunc, mutation_op, {"iter":100, "temp_init":30, "alpha":0.99})
    elif alg_name == "DE":
        search_strat = DE(objfunc, OperatorReal("DE/best/1", {"F":0.2, "Cr":0.5, "P":0.11}), {"popSize":100})
    else:
        print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
        exit()
    

    alg = GeneralSearch(search_strat, params)


    # Optimize with display of image
    time_start = time.process_time()
    real_time_start = time.time()
    display_timer = time.time()

    alg.initialize()

    while not alg.ended:
        # process GUI events and reset screen
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit(0)
            src.fill('#000000')
        
        alg.step(time_start=real_time_start)

        if alg.verbose and time.time() - display_timer > alg.v_timer:
            alg.step_info(real_time_start)
            display_timer = time.time()
        
        if display:
            img_flat = alg.best_solution()[0]
            render(img_flat.reshape(image_shape + [3]).astype(np.int32), display_dim, src)
            pygame.display.update()
    
    alg.real_time_spent = time.time() - real_time_start
    alg.time_spent = time.process_time() - time_start
    img_flat = alg.best_solution()[0]
    image = img_flat.reshape(image_shape + [3])
    render(image, display_dim, src)
    alg.display_report()
    save_to_image(image, f"{img_name}_{image_shape[0]}x{image_shape[1]}_{alg_name}.png")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest='alg', help='Specify an algorithm')
    parser.add_argument("-i", "--image", dest='img', help='Specify an image as reference')
    args = parser.parse_args()

    algorithm_name = "SA"
    img_file_name = "images/cat.png"

    if args.alg:
        algorithm_name = args.alg
    
    if args.img:
        img_file_name = args.img
   
    run_algorithm(alg_name = algorithm_name, img_file_name = img_file_name)


if __name__ == "__main__":
    main()