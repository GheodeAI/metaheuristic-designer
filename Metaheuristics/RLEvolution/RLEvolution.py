from ..Metaheuristic import *
from .EvolSarsa import *
from .EvolQLearning import *

class RLEvolution(Metaheuristic):
    def __init__(self, objfunc, operators, params):
        super().__init__("RLevol", objfunc, params)

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.operators = operators
        self.population = EvolQLearning(objfunc, operators, params)
        # self.population = EvolSarsa(objfunc, operators, params)
    
    def restart(self):
        super().restart()

        self.population = EvolQLearning(self.objfunc, self.operators, self.params)
        # self.population = EvolSarsa(self.objfunc, self.operators, self.params)

    """
    One step of the algorithm
    """
    def step(self, progress):
        self.population.update(progress)
        
        self.population.evolve()
        
        self.population.replace()

        self.population.step(progress)
        
        super().step(progress)
    
    """
    Shows a summary of the execution of the algorithm
    """
    def display_report(self, show_plots=True):
        # Print Info
        print("Number of generations:", len(self.history))
        print("Real time spent: ", round(self.real_time_spent, 5), "s", sep="")
        print("CPU time spent: ", round(self.time_spent, 5), "s", sep="")
        print("Number of fitness evaluations:", self.objfunc.counter)
        
        best_fitness = self.population.best_solution()[1]
        print("Best fitness:", best_fitness)

        if show_plots:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
            fig.suptitle(f"{self.name}")
            plt.subplot(1, 2, 1)


            # Plot fitness history            
            plt.plot(np.arange(len(self.history)), self.history, "blue")
            plt.xlabel("generations")
            plt.ylabel("fitness")
            plt.title(f"{self.name} fitness")

            plt.subplot(1, 2, 2)

            op_res = scipy.special.softmax(-self.population.q_table, axis=1)
            op_res = op_res + 0.1/op_res.shape[1]
            op_res = op_res/op_res.sum()

            # data to plot
            n_groups = op_res.shape[0]

            # create plot
            index = np.arange(n_groups)
            bar_width = 0.9/(op_res.shape[1])
            opacity = 0.8

            for i in range(op_res.shape[1]):
                plt.bar(index + i*bar_width, op_res[:, i], bar_width,
                    alpha=opacity,
                    label=self.operators[i].name)

            plt.xlabel('Operator')
            plt.ylabel('Eval')
            plt.title('Eval of each operator Softmax')
            plt.xticks(index + bar_width, ['State' +  str(i) for i in range(op_res.shape[0])], rotation=345)
            plt.legend()
            plt.show()