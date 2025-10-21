import seaborn as sns
from matplotlib import patches as mpatches, pyplot as plt

from utils import load_all


def dataset_summaries():
    data = load_all(1)
    data = data[data["shift"].isin(["ind_val", "ind_test", "train"])]
    data = data[data["feature_name"]=="energy"] #random
    data = data[data["Dataset"]!="Polyp"] #polyp is special

    print(data)
    g = sns.FacetGrid(data, col="Dataset", height=3, aspect=1.5, sharex=False, sharey=False, col_wrap=2)
    g.map_dataframe(sns.countplot, x="class")
    for ax in g.axes.flat:
        dataset_name_for_ax = ax.get_title().split(" = ")[-1]
        ax.set_title(dataset_name_for_ax)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        # Set x-ticks to be the class names
        ax.set_xticklabels([])
    plt.savefig("class_distribution.pdf")
    plt.show()


def accuracy_table():
    df = load_all(1, samples=1000, prefix="coarse_data")
    print(df["shift"].unique())
    df = df[df["shift"]!="noise"]
    accs = df.groupby(["Dataset", "shift"])["correct_prediction"].mean().reset_index()
    print(accs)
    return accs
