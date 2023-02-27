from ..Metaheuristic import *
from .CoralPopulation import *


class CRO_SL(Metaheuristic):
    """
    Coral reef optimization with substrate layers
    """
    
    def __init__(self, objfunc, substrates, params):
        """
        Constructor of the CRO algorithm.

        Parameters:
            popSize: maximum number of corals in the reef        [Integer,          100  recomended]
            rho: percentage of initial ocupation of the reef     [Real from 0 to 1, 0.6  recomended]
            Fb: broadcast spawning proportion                    [Real from 0 to 1, 0.98 recomended]
            Fd: depredation proportion                           [Real from 0 to 1, 0.2  recomended]
            Pd: depredation probability                          [Real from 0 to 1, 0.9  recomended]
            k: maximum attempts for larva setting                [Integer,          4    recomended]
            K: maximum amount of corals with duplicate solutions [Integer,          20   recomended]

            group_subs: evolve only within the same substrate or with the whole population

            dynamic: change the size of each substrate
            dyn_method: which value to use for the evaluation of the substrates
            dyn_metric: how to process the data from the substrate for the evaluation
            dyn_steps: number of times the substrates will be evaluated
            prob_amp: parameter that makes probabilities more or less extreme with the same data

        """
        super().__init__("DPCRO-SL", objfunc, params)

        # Dynamic parameters
        self.dynamic = params["dynamic"] if "dynamic" in params else True
        self.dyn_method = params["dyn_method"] if "dyn_method" in params else "fit"

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.substrates = substrates
        self.population = CoralPopulation(objfunc, substrates, params)


    def restart(self):
        """
        Resets the data of the CRO algorithm
        """

        super().restart()

        self.population = CoralPopulation(self.objfunc, self.substrates, self.params)

    
    def step(self, progress, depredate=True, classic=False):
        """
        One step of the algorithm
        """

        if not classic:
            self.population.generate_substrates(progress)

        larvae = self.population.evolve_with_substrates()
        
        self.population.larvae_setting(larvae)

        if depredate:
            self.population.extreme_depredation()
            self.population.depredation()
        
        self.population.step(progress)
        
        super().step(progress)
    
    
    def local_search(self, operator, n_ind, iterations=100):
        """
        Performs local search with a given operator
        """
        if self.verbose:
            print(f"Starting local search, {n_ind} individuals searching {iterations} neighbours each.")
        
        self.population.local_search(operator, n_ind, iterations)
        return self.population.best_solution()
    
    
    def safe_optimize(self):
        """
        Execute the algorithm with early stopping
        """

        result = (np.array([]), 0)
        try:
            result = self.optimize()
        except KeyboardInterrupt:
            print("stopped early")
            self.save_solution(file_name="stopped.csv")
            self.display_report(show_plots=False, save_figure=True, figure_name="stopped.eps")
            exit(1)
        
        return result

    
    def optimize_classic(self):
        """
        Execute the classic version of the algorithm
        """

        gen = 0
        time_start = time.process_time()
        real_time_start = time.time()
        display_timer = time.time()

        self.population.generate_random()
        self.step(0, depredate=False)
        while not self.stopping_condition(gen, real_time_start):
            prog = self.progress(gen, real_time_start)
            self.step(prog, depredate=True, classic=True)
            gen += 1
            if self.verbose and time.time() - display_timer > self.v_timer:
                self.step_info(gen, real_time_start)
                display_timer = time.time()
                
        self.real_time_spent = time.time() - real_time_start
        self.time_spent = time.process_time() - time_start
        return self.population.best_solution()

    
    def step_info(self, gen, start_time):
        """
        Displays information about the current state of the algotithm
        """
        
        super().step_info(gen, start_time)

        if self.dynamic:
            print(f"\tSubstrate probability:")
            subs_names = [i.name for i in self.substrates]
            weights = [round(i, 6) for i in self.population.substrate_weight]
            adjust = max([len(i) for i in subs_names])
            for idx, val in enumerate(subs_names):
                print(f"\t\t{val}:".ljust(adjust+3, " ") + f"{weights[idx]}")
        print()
    
    
    def display_report(self, show_plots=True, save_figure=False, figure_name="fig.eps"):
        """
        Shows a summary of the execution of the algorithm
        """

        if save_figure:
            if not os.path.exists("figures"):
                os.makedirs("figures")

        if self.dynamic:
            self.display_report_dyn(show_plots, save_figure, figure_name)
        else:
            self.display_report_nodyn(show_plots, save_figure, figure_name)

    
    def display_report_dyn(self, show_plots=True, save_figure=False, figure_name="fig.eps"):
        """
        Version of the summary for the dynamic variant of the algorithm
        """

        factor = 1
        if self.objfunc.opt == "min" and self.dyn_method != "success":
            factor = -1

        # Print Info
        print("Number of generations:", len(self.history))
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", self.objfunc.counter)
        print(f"\tSubstrate probability:")
        subs_names = [i.name for i in self.substrates]
        weights = [round(i, 6) for i in self.population.substrate_weight]
        adjust = max([len(i) for i in subs_names])
        for idx, val in enumerate(subs_names):
            print(f"\t\t{val}:".ljust(adjust+3, " ") + f"{weights[idx]}")
        
        best_fitness = self.population.best_solution()[1]
        print("Best fitness:", best_fitness)

        
        if save_figure:
            # Plot fitness history            
            plt.plot(self.history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("DPCRO_SL fitness")
            plt.savefig(f"figures/fit_{figure_name}")
            plt.cla()
            plt.clf()
            
            m = np.array(self.population.substrate_history)[1:].T
            for i in m:
                plt.plot(factor * i)
            plt.legend([i.name for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("Evaluation of each substrate")
            plt.savefig(f"figures/eval_{figure_name}")
            plt.cla()
            plt.clf()
            

            prob_data = np.array(self.population.substrate_w_history).T
            plt.stackplot(range(prob_data.shape[1]), prob_data, labels=[i.name for i in self.substrates])
            plt.legend([i.name for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("probability")
            plt.title("Probability of each substrate")
            plt.savefig(f"figures/prob_{figure_name}")
            plt.cla()
            plt.clf()


            plt.plot(prob_data.T)
            plt.legend([i.name for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("Evaluation of each substrate")
            plt.savefig(f"figures/subs_{figure_name}")
            plt.cla()
            plt.clf()

            plt.close("all")
            
        if show_plots:
            # Plot fitness history            
            fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10,10))
            fig.suptitle("DPCRO_SL")
            plt.subplot(2, 2, 1)
            
            plt.plot(self.history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("DPCRO_SL fitness")

            
            plt.subplot(2, 2, 2)
            m = np.array(self.population.substrate_history)[1:].T
            for i in m:
                plt.plot(factor * i)
            plt.legend([i.name for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("Evaluation of each substrate")

            plt.subplot(2, 1, 2)
            prob_data = np.array(self.population.substrate_w_history).T
            plt.stackplot(range(prob_data.shape[1]), prob_data, labels=[i.name for i in self.substrates])
            plt.legend([i.name for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("probability")
            plt.title("Probability of each substrate")

            plt.show()

    
    def display_report_nodyn(self, show_plots=True, save_figure=False, figure_name="fig.eps"):
        """
        Version of the summary for the non-dynamic variant of the algorithm
        """

        factor = 1
        if self.objfunc.opt == "min":
            factor = -1

        # Print Info
        print("Number of generations:", len(self.history))
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", self.objfunc.counter)
        print(f"\tSubstrate probability:")
        subs_names = [i.name for i in self.substrates]
        weights = [round(i, 6) for i in self.population.substrate_weight]
        adjust = max([len(i) for i in subs_names])
        for idx, val in enumerate(subs_names):
            print(f"\t\t{val}:".ljust(adjust+3, " ") + f"{weights[idx]}")
        
        best_fitness = self.population.best_solution()[1]
        print("Best fitness:", best_fitness)

        if save_figure:            
            plt.plot(self.history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("PCRO_SL fitness")
            plt.savefig(f"figures/fit_{figure_name}")
            plt.cla()
            plt.clf()

            
            m = np.array(self.population.substrate_history)[1:].T
            for i in m:
                plt.plot(factor * i)
            plt.legend([i.name for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("Evaluation of each substrate")
            plt.savefig(f"figures/eval_{figure_name}")
            plt.cla()
            plt.clf()
            plt.close("all")

        if show_plots:
            # Plot fitness history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
            fig.suptitle("CRO_SL")
            plt.subplot(1, 2, 1)
            
            plt.plot(self.history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("PCRO_SL fitness")

            plt.subplot(1, 2, 2)
            m = np.array(self.population.substrate_history)[1:].T
            for i in m:
                plt.plot(factor * i)
            plt.legend([i.name for i in self.substrates])
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title("Evaluation of each substrate")

            if save_figure:
                plt.savefig(f"figures/{figure_name}")

            plt.show()
