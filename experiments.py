from pra import *
import seaborn as sns

def uniform_bernoulli(data, estimator=BernoulliEstimator, load=True):
    if load:
        results = pd.read_csv("risk_uniform_bernoulli.csv")
    else:
        x = np.linspace(0,1,11)
        result_list = []
        for i in x:
            sim = SystemSimulator(data, ood_test_shift=ENDOCV, maximum_loss=0.5, estimator=estimator)
            results = sim.uniform_rate_sim(i, 10000)
            re, ndsdre, tr, ndsdtr = results.mean()
            result_list.append({"p": i, "Risk":re, "DSD":True, "True Risk": False, "error":np.abs(tr-re)})
            result_list.append({"p": i, "Risk":tr, "DSD":True, "True Risk": True, "error":np.abs(tr-re)})
            result_list.append({"p": i, "Risk":ndsdre, "DSD":False, "True Risk": False, "error":np.abs(ndsdre-ndsdtr)})
            result_list.append({"p": i, "Risk":ndsdtr, "DSD":False, "True Risk": True, "error":np.abs(ndsdre-ndsdtr)})
        results = pd.DataFrame(result_list)
        results.to_csv("risk_uniform_bernoulli.csv")
        results = pd.read_csv("risk_uniform_bernoulli.csv")


    results = results[results["DSD"]==True]
    sns.lineplot(results, x="p", y="error")
    plt.savefig("risk_uniform_bernoulli.eps")
    plt.show()
    return results

def single_run(data, estimator=BernoulliEstimator):
    sim = SystemSimulator(data, ood_test_shift=ENDOCV, maximum_loss=0.5, estimator=estimator)
    results = sim.uniform_rate_sim(0.5, 10000)
    print(results.mean())
    sim = SystemSimulator(data, ood_test_shift=CVCCLINIC, maximum_loss=0.5, estimator=estimator)
    results = sim.uniform_rate_sim(0.5, 10000)
    print(results.mean())


if __name__ == '__main__':
    data = load_pra_df("knn", batch_size=10, samples=1000)
    single_run(data)
    # uniform_bernoulli(data, load = False)