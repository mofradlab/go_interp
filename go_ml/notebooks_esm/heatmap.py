import matplotlib.pyplot as plt
import seaborn as sns
import torch

%matplotlib qt
def plot_heatmap(model):
    # sequence = protein_df["Sequence"]
    # go_ind = protein_df["GOTermIndex"]
    # length = len(sequence)
    # embeddings = ["aromaticity", "isoelectric", "molecular_weight"]

    weights_unflattened = torch.unflatten(model.get_weights(), 1, (model.input_size[1], model.input_size[2])).squeeze(0).T
    importances = model.get_importance_abs()

    print(weights_unflattened.shape)
    print(importances.shape)
    # Create a Matplotlib Figure and Axes objects using subplot
    fig, axes = plt.subplots(5, 1, figsize=(50, 2))  # Creating 1 row, 2 columns of subplots

# Plotting the first heatmap
    sns.heatmap(weights_unflattened, ax=axes[0], annot=True)
    axes[0].set_title('Weights Matrix')

# Plotting the second heatmap
    sns.heatmap(importances, ax=axes[1], annot=True)
    axes[1].set_title('Importances')

# Adjust layout to prevent overlap of titles and labels
    plt.tight_layout()

# Show the plot
    plt.show()