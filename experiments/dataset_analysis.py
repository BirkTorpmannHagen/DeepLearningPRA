import seaborn as sns
from matplotlib import patches as mpatches, pyplot as plt

from utils import load_polyp_data, load_all


def iou_distribution():

    df = load_polyp_data()
    df = df[df["shift"] != "train"]
    df = df[df["shift"] != "ind_val"]
    df.replace({"ind_test":"Kvasir"}, inplace=True)
    palette = sns.color_palette("tab10", n_colors=df["Model"].nunique())
    g = sns.FacetGrid(df, col="shift", palette=palette)
    # g = sns.FacetGrid(df, col="shift")
    g.map_dataframe(sns.kdeplot, x="IoU", hue="Model", clip=(0, 1), fill=True, multiple="stack", common_norm=False)
    # manually extract legend from one of the axes
    # g._legend.remove()  # in case it partially shows
    df.replace({"deeplabv3plus":"DeepLabV3+", "unet":"UNet++", "segformer":"SegFormer"}, inplace=True)
    model_names = sorted(df["Model"].unique())
    colors = sns.color_palette("tab10", n_colors=len(model_names))
    color_dict = dict(zip(model_names, colors))
    patches = [mpatches.Patch(color=color_dict[m], label=m) for m in model_names]
    plt.legend(handles=patches, title="Model", frameon=True,
               loc="best", ncol=1)
    plt.savefig("IoU_distributions.pdf")
    plt.show()


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
    df = load_all(1, samples=1000, prefix="final_data")
    df = df[df["shift"]!="noise"]
    accs = df.groupby(["Dataset", "shift"])["correct_prediction"].mean().reset_index()
    print(accs)
    return accs
