import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, export_graphviz
import graphviz

# Load the trained model
model = joblib.load('../models/iris_model.joblib')

# Print the model to get a basic overview
print(model)

# Print model parameters
print("\nModel Parameters:")
print(model.get_params())

# If it’s a RandomForestClassifier, we can inspect more details
if isinstance(model, RandomForestClassifier):
    print("\nFeature Importances:")
    print(model.feature_importances_)

    # Print details of individual trees (for example, the first tree)
    print("\nFirst tree details:")
    tree = model.estimators_[0]
    print(tree)

    # Visualize the first tree
    plt.figure(figsize=(10, 10))
    plot_tree(tree, filled=True, feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'])
    plt.show()

    # Alternatively, export the tree in DOT format and render using graphviz
    # dot_data = export_graphviz(tree, out_file=None, filled=True, feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'])
    # graph = graphviz.Source(dot_data)
    # graph.render("first_tree")
