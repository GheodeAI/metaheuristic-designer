conda activate bench-mhd
parallel python run_GA.py -o {1} -d {2} --ngen 300 -f result_GA-pygad-{1}-d{2}.csv ::: 'Sphere' 'Rastrigin' 'Rosenbrock' 'Weierstrass' ::: 1 2 3 10 100
