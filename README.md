# Machine Learning Repository

A comprehensive collection of machine learning projects, tutorials, experiments, and learning sessions covering various ML algorithms, datasets, and real-world applications.

## Repository Structure

```
.
├── notebooks/              # Jupyter notebooks organized by purpose
│   ├── tutorials/         # Learning materials and algorithm implementations
│   │   ├── algorithms/    # Linear regression, MNIST, and other algorithms
│   │   ├── backpropagation/
│   │   ├── computer-vision/
│   │   └── convolutional-neural-network/
│   ├── examples/          # Dataset examples and demonstrations
│   │   ├── breastCancer.ipynb
│   │   ├── CaliforniaHousing.ipynb
│   │   └── SVM(irisdataset).ipynb
│   ├── experiments/       # Test notebooks and custom implementations
│   │   ├── CSV_to_dataset_keras.ipynb
│   │   └── Keras_custom_model.ipynb
│   └── visualization/     # Data visualization notebooks and resources
│       └── data_visualization.ipynb
├── projects/              # Production-ready ML projects with Flask APIs
│   ├── breast-cancer-project/
│   ├── california-housing-project/
│   ├── diabetes-project/
│   ├── irisFeature-project/
│   ├── music-genre-generation-project/
│   ├── student-grade-project/
│   ├── student_performance_index/
│   ├── videoGame-project/
│   └── wine-project/
└── sessions/              # Learning sessions and practice work
    ├── 02-02-2026/
    ├── 19-01-2026/
    └── 22-01-2026_Classification/
```

## Getting Started

1. **Learning**: Navigate to `notebooks/tutorials/` for algorithm implementations and learning materials
2. **Examples**: Check `notebooks/examples/` for dataset-specific demonstrations (Breast Cancer, California Housing, Iris SVM)
3. **Experiments**: Explore `notebooks/experiments/` for custom Keras models and data processing techniques
4. **Projects**: Browse `projects/` for complete ML applications with APIs and demos
5. **Sessions**: Review `sessions/` for dated learning sessions and classification work

## Projects

Each project folder typically contains:
- **Training scripts** (`train.py`) - Model training and evaluation
- **Flask API** (`app.py`) - REST API for model predictions
- **Trained models** (`model/`) - Serialized model files
- **Demo applications** (`demo/`, `nodeApp/`) - Frontend interfaces for testing

### Available Projects
- **Breast Cancer Detection** - Classification model for cancer diagnosis
- **California Housing** - Regression model for housing price prediction
- **Diabetes Prediction** - Healthcare prediction model
- **Iris Feature Classification** - Classic iris dataset classification
- **Music Genre Generation** - Audio/music classification
- **Student Grade Prediction** - Educational performance prediction
- **Student Performance Index** - Academic performance analysis
- **Video Game Analysis** - Gaming data analysis
- **Wine Quality** - Wine classification/regression

## Notebooks

### Tutorials
Comprehensive learning materials covering:
- Algorithm implementations (linear regression, neural networks, etc.)
- Backpropagation fundamentals
- Computer vision techniques
- Convolutional neural networks (CNNs)

### Examples
Real-world dataset implementations:
- Breast cancer classification using various algorithms
- California housing price prediction
- Support Vector Machines (SVM) on iris dataset

### Experiments
Custom implementations and explorations:
- CSV to Keras dataset conversion
- Custom Keras model architectures

### Visualization
Data analysis and visualization techniques for ML datasets

## Sessions

Dated learning sessions containing practice work, experiments, and specific topic explorations (e.g., classification techniques, recommendation systems)
