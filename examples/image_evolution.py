from pyevolcomp import ObjectiveFunc, ParentSelection, SurvivorSelection, ParamScheduler
from pyevolcomp.SearchMethods import GeneralSearch, MemeticSearch
from pyevolcomp.Operators import OperatorReal, OperatorInt, OperatorBinary
from pyevolcomp.Algorithms import *
from pyevolcomp.Decoders import ImageDecoder
from pyevolcomp.benchmarks import * 

import pygame
import time
import numpy as np
import cv2
import os
from copy import deepcopy
from PIL import Image

# import matplotlib
# matplotlib.use("Gtk3Agg")

import argparse


def render(image, display_dim, src):
    texture = cv2.resize(image.transpose([1,0,2]), display_dim, interpolation = cv2.INTER_NEAREST)
    pygame.surfarray.blit_array(src, texture)
    pygame.display.flip()

def save_to_image(image, img_name="result.png"):
    if not os.path.exists('./examples/results/'):
        os.makedirs('./examples/results/')
    filename = './examples/results/' + img_name
    Image.fromarray(image.astype(np.uint8)).save(filename)

def run_algorithm(alg_name, img_file_name, memetic):
    params = {
        # General
        "stop_cond": "time_limit",
        "time_limit": 60.0,
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

    decoder = ImageDecoder(image_shape, color=True)

    reference_img = Image.open(img_file_name)
    img_name = img_file_name.split("/")[-1]
    img_name = img_name.split(".")[0]
    objfunc = ImgApprox(image_shape, reference_img, img_name=img_name, decoder=decoder)
    # objfunc = ImgEntropy(image_shape, 256, decoder=decoder)
    # objfunc = ImgExperimental(image_shape, reference_img, img_name=img_name, decoder=decoder)

    mutation_op = OperatorInt("MutRand", {"method": "Cauchy", "F":15, "N":20})
    cross_op = OperatorReal("Multicross", {"Nindiv": 4})
    parent_sel_op = ParentSelection("Best", {"amount": 15})
    selection_op = SurvivorSelection("Elitism", {"amount": 10})

    if alg_name == "HillClimb":
        search_strat = HillClimb(mutation_op)
    elif alg_name == "LocalSearch":
        search_strat = LocalSearch(mutation_op, {"iters":20})
    elif alg_name == "ES":
        selection_op = SurvivorSelection("(m+n)")
        search_strat = ES(mutation_op, cross_op, parent_sel_op, selection_op, {"popSize":100, "offspringSize":500})
    elif alg_name == "GA":
        search_strat = GA(mutation_op, cross_op, parent_sel_op, selection_op, {"popSize":100, "pcross":0.9, "pmut":0.15})
    elif alg_name == "HS":
        search_strat = HS({"HMS":100, "HMCR":0.8, "BW":0.5, "PAR":0.2})
    elif alg_name == "SA":
        search_strat = SA(mutation_op, {"iter":100, "temp_init":1, "alpha":0.9975})
    elif alg_name == "DE":
        de_op = OperatorReal("DE/best/1", {"F":0.2, "Cr":0.3, "P":0.11})
        search_strat = DE(de_op, {"popSize":100})
    elif alg_name == "PSO":
        search_strat = PSO({"popSize":100, "w":0.7, "c1":1.5, "c2":1.5})
    elif alg_name == "NoSearch":
        search_strat = NoSearch({"popSize":100})
    else:
        print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
        exit()
    
    if memetic:
        local_search = LocalSearch(OperatorInt("MutRand", {"method": "Uniform", "Low":-3, "Up":-3, "N":3}), {"iters":10})
        alg = MemeticSearch(search_strat, local_search, ParentSelection("Best", {"amount": 10}), params)
    else:
        alg = GeneralSearch(objfunc, search_strat, params=params)

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
            render(decoder.decode(img_flat), display_dim, src)
            pygame.display.update()
    
    alg.real_time_spent = time.time() - real_time_start
    alg.time_spent = time.process_time() - time_start
    img_flat = alg.best_solution()[0]
    image = img_flat.reshape(image_shape + [3])
    if display:
        render(image, display_dim, src)
    alg.display_report(show_plots=True)
    save_to_image(image, f"{img_name}_{image_shape[0]}x{image_shape[1]}_{alg_name}.png")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest='alg', help='Specify an algorithm')
    parser.add_argument("-i", "--image", dest='img', help='Specify an image as reference')
    parser.add_argument("-m", "--memetic", dest='mem', action="store_true", help='Specify an algorithm')
    args = parser.parse_args()

    algorithm_name = "SA"
    img_file_name = "images/cat.png"
    mem = False

    if args.alg:
        algorithm_name = args.alg
    
    if args.img:
        img_file_name = args.img
    
    if args.mem:
        mem = True
   
    run_algorithm(alg_name = algorithm_name, img_file_name = img_file_name, memetic=mem)


if __name__ == "__main__":
    main()