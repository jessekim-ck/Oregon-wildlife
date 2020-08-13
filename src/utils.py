from matplotlib import pyplot as plt


def draw_cost_curve(train_cost_dict, test_cost_dict, model_name):
    plt.figure()
    plt.plot(*zip(*sorted(train_cost_dict.items())), color="blue", label="train")
    plt.plot(*zip(*sorted(test_cost_dict.items())), color="red", label="test")
    plt.title("Cost plot")
    plt.legend()
    plt.savefig(f"results/plots/{model_name}.png")
    plt.close()
