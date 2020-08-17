from matplotlib import pyplot as plt


def draw_cost_curve(train_cost_dict, test_cost_dict, accuracy_dict, model_name):
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    plt.plot(*zip(*sorted(train_cost_dict.items())), color="blue", label="train")
    plt.plot(*zip(*sorted(test_cost_dict.items())), color="red", label="test")
    plt.title("Cost plot")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(*zip(*sorted(accuracy_dict.items())), color="red")
    plt.title("Test accuracy")

    plt.savefig(f"results/plots/{model_name}.png")
    plt.close()
