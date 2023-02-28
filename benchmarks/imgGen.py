import sys
sys.path.append("../..")

from PyEvolAlg import *
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
    params_base = {
        # Population-based
        "popSize": 100,

        # Coral reef optimization
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 0.2,
        "Pd": 0.9,
        "k": 3,
        "K": 20,

        ## Dynamic CRO-SL
        "group_subs": False,

        "dynamic": True,
        "dyn_method": "fitness",
        "dyn_metric": "best",
        "dyn_steps": 100,
        "prob_amp": 0.1,

        # Genetic algorithm
        "pmut": 0.2,
        "pcross":0.9,

        # Evolution strategy
        "offspringSize":500,

        # Particle swarm optimization
        "w": 0.729,
        "c1": 1.49445,
        "c2": 1.49445,

        # Reinforcement learning based search
        "discount": 0.6,
        "alpha": 0.7,
        "eps": 0.1,
        "nstates": 5,

        "sel_exp": 2,

        # Harmony search
        "HMCR": 0.9,
        "PAR" : 0.3,
        "BN" : 1,

        # Hill Climb
        #"p": [0.1, -0.2],
        "p": 0.01,

        # Simulated annealing
        "iter": 100,
        "temp_init": 5,
        "alpha" : 0.995,

        # General
        "stop_cond": "neval",
        "time_limit": 20.0,
        "ngen": 1000,
        "neval": 6e5,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": -1
    }
    
    params = ParamScheduler("Linear", params_base)

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
    # objfunc = ImgStd(image_shape, "min")
    # objfunc = ImgEntropy(image_shape, 256, "min")
    # objfunc = ImgExperimental(image_shape, reference_img)

    operators = [
        OperatorInt("MutSample", {"method": "Uniform", "Low":0, "Up":256, "N":10}),
        OperatorInt("MutRand", {"method": "Uniform", "Low":-10, "Up":10, "N":10}),
        OperatorInt("Multipoint")
    ]
    
    # mutation_op = OperatorInt("Xor", {"N":20})
    # mutation_op = OperatorInt("Gauss", {"F": 1})
    #mutation_op = OperatorInt("MutRand", {"method": "Uniform", "Low":-10, "Up":10, "N":100})
    #mutation_op = OperatorInt("MutSample", ParamScheduler("Linear", {"method": "Uniform", "Low":0, "Up":256, "N":[1000,50]}))
    mutation_op = OperatorInt("MutSample", {"method": "Uniform", "Low":0, "Up":256, "N":1000})
    cross_op = OperatorInt("2point")
    #parent_select_op = ParentSelection("Tournament", {"amount": 20, "p":0.1})
    parent_select_op = ParentSelection("Nothing")
    replace_op = SurvivorSelection("CondElitism", {"amount": 10})

    if alg_name == "CRO_SL":
        alg = CRO_SL(objfunc, operators, params)
    elif alg_name == "GA":
        alg = Genetic(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
    elif alg_name == "ES":
        alg = ES(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
    elif alg_name == "DE":
        de_op = OperatorReal("DE/current-to-best/1", {"F":0.9, "Cr":0.8})
        alg = DE(objfunc, de_op, SurvivorSelection("One-to-one"), params)
    elif alg_name == "PSO":
        alg = PSO(objfunc, params)
    elif alg_name == "RLevol":
        alg = RLEvolution(objfunc, operators, params)
    elif alg_name == "HS":
        alg = HS(objfunc, OperatorReal("Gauss", {"F":params["BN"]}), OperatorReal("RandSample", {"method":"Gauss", "F":params["BN"]}), params)
    elif alg_name == "In-HS":
        operators_mut_InHS = [
            OperatorReal("Gauss", {"F":params["BN"]}),
            OperatorReal("Cauchy", {"F":params["BN"]/2}),
            OperatorReal("Laplace", {"F":params["BN"]}),
        ]
        alg = InHS(objfunc, operators_mut_InHS, OperatorReal("RandSample", {"method":"Gauss", "F":params["BN"]}), params)
    elif alg_name == "HillClimb":
        alg = HillClimb(objfunc, mutation_op, params)
    elif alg_name == "SA":
        alg = SimAnn(objfunc, mutation_op, params)
    else:
        print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
        exit()



    # Optimize with display of image
    gen = 0
    time_start = time.process_time()
    real_time_start = time.time()
    display_timer = time.time()

    alg.population.generate_random()

    while not alg.stopping_condition(gen, real_time_start):
        # process GUI events and reset screen
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit(0)
            src.fill('#000000')
        
        prog = alg.progress(gen, real_time_start)
        alg.step(prog)
        gen += 1
        if alg.verbose and time.time() - display_timer > alg.v_timer:
            alg.step_info(gen, real_time_start)
            display_timer = time.time()
        
        if display:
            img_flat = alg.population.best_solution()[0]
            render(img_flat.reshape(image_shape + [3]).astype(np.int32), display_dim, src)
            pygame.display.update()
    
    alg.real_time_spent = time.time() - real_time_start
    alg.time_spent = time.process_time() - time_start
    img_flat = alg.population.best_solution()[0]
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