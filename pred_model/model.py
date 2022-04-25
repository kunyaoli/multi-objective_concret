from pymoo.core.problem import ElementwiseProblem
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize


class MyProblem(ElementwiseProblem):

    def __init__(self,var,xl_list,xu_list,clf, UCS_target,concrete,concrete_var):
        super().__init__(n_var=var,
                         n_obj=2,
                         n_constr=8,
                         xl=np.array(xl_list), #[320.0, 0.35, 0.0, 0.0, 0.25, 0.35, 2350.0, 42.5]
                         xu=np.array(xu_list),#[480.0, 1.0, 0.45, 0.65, 0.36, 0.45, 2450.0, 70.0]
                         )
        self.clf = clf
        self.UCS_target = UCS_target
        self.concrete = concrete
        self.concrete_var = concrete_var

    def _evaluate(self, x, out, *args, **kwargs):
        UCS = -self.clf.predict(x.reshape(1, -1))

        # definition
        x_1 = x[0]
        x_2 = x[1]
        x_3 = x[2]
        x_4 = x[3]
        x_5 = x[4]
        x_6 = x[5]
        x_7 = x[6]
        x_8 = x[7]

        COST = (((x_1 * x_2) * 630 + (x_1 * x_3) * 285 + (x_1 * x_4) * 500 + (
                    x_6 * (x_7 - x_1 - x_1 * x_5 - 0.01 * x_1)) * 200 + (
                             (1 - x_6) * (x_7 - x_1 - x_1 * x_5 - 0.01 * x_1) * 175) + (x_1 * 0.01 * 3100))) / 1000

        g1 = x_2 + x_3 + x_4 - 1
        g2 = 0.99 - (x_2 + x_3 + x_4)
        g3 = -(self.UCS_target+self.concrete_var + UCS)  # UCS DATA
        g4 = self.UCS_target-self.concrete_var + UCS
        g5 = x_3 + x_4 - 0.65
        g6 = -(x_3 + x_4)
        g7 = x_8 - (self.concrete+self.concrete_var)
        g8 = (self.concrete-self.concrete_var) - x_8

        out["F"] = [UCS, COST]
        out["G"] = [g1, g2, g3, g4, g5, g6, g7, g8]

if __name__ == '__main__':
    import pandas as pd
    from sklearn.externals import joblib
    from pymoo.factory import get_termination

    data = pd.read_csv("../data/constrain.csv")
    xl_list = data.values[:, 1]
    xu_list = data.values[:, 2]
    clf = joblib.load(filename="./RF.pkl")
    target_concrete = 55
    cement_strength = 50
    concrete_val = 2
    problem = MyProblem(var=data.shape[0],xl_list=xl_list, xu_list=xu_list, clf=clf,
                        UCS_target=target_concrete, concrete=cement_strength, concrete_var=concrete_val)

    pf = problem.pareto_front(use_cache=False, flatten=True)
    algorithm = NSGA2(
        pop_size=200,
        n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=10),
        eliminate_duplicates=True
    )


    termination = get_termination("n_gen", 600)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=False)

    X, F = res.opt.get("X", "F")
    #hist = res.history
    #print(len(hist))
