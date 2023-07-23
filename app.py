import streamlit as st
import pandas as pd
from sklearn.externals import joblib
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pred_model.model import MyProblem
from pymoo.factory import get_termination
from pic_plot.plot_parallel import parallel
from pic_plot.plt_scatter import plot_sacatter

st.title("Concrete multi-label optimal")
st.write("Kunyao Li")
st.write("CCCC Second Harbor Engineering Company LTD")
st.write("likunyao@ccccltd.cn")
st.write("---")
target_concrete = st.number_input(label="Concrete Compressive Strength :", min_value=0, value=45)
cement_strength = st.number_input(label="Cement Strength :", min_value=0, value=42)
concrete_var = st.number_input(label="Var :", min_value=1, value=2)
termination_num = st.number_input(label="termination_num :", min_value=0, value=500)
constrain_file = st.file_uploader('Upload your own constrain data')
agree = st.checkbox('Use example file', help="Built-in constraints")
constrain_df = None

if agree:
    constrain_df = pd.read_csv("./data/constrain.csv")
    #st.table(constrain_df)
if constrain_file:
    constrain_df = pd.read_csv(constrain_file)
    #st.dataframe(constrain_df)

if st.button('run'):
    with st.spinner('Wait for it...'):
        xl_list = constrain_df.values[:, 1]
        xu_list = constrain_df.values[:, 2]
        clf = joblib.load(filename="pred_model/RF.pkl")
        problem = MyProblem(var=constrain_df.shape[0],xl_list=xl_list, xu_list=xu_list, clf=clf,
                            UCS_target= target_concrete,concrete=cement_strength,concrete_var= concrete_var)
        pf = problem.pareto_front(use_cache=False, flatten=True)
        algorithm = NSGA2(
            pop_size=300,
            n_offsprings=10,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=10),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", termination_num)
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=False)
        X, F = res.opt.get("X", "F")
        hist = res.history

        dataframe = pd.DataFrame(X)
        dataframe["COST"] = F[:, 1]
        dataframe["UCS"] = F[:, 0] * -1
        #st.dataframe(dataframe)


        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(dataframe)
        st.download_button(
            label="Press to Download",
            data=csv,
            file_name="result.csv",
            mime="text/csv",
        )
        #st.text(dataframe.columns)
        #st.text(dataframe[0])
        fig = parallel(dataframe)
        st.plotly_chart(fig)

        fig2 = plot_sacatter(dataframe)
        st.plotly_chart(fig2)
    st.success('Done!')

else:
    pass
