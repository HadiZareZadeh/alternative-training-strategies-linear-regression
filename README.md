# Exploring Alternative Training Strategies for Linear Regression

## Project Description

This project investigates alternative training strategies for linear regression, moving beyond standard mini-batch gradient descent. We explore innovative approaches to training that challenge conventional wisdom about how optimization should work. The project examines data-centric training strategies and adaptive optimization techniques.

**Why This Project Matters**:
- **Training Innovation**: Explores cutting-edge ideas in optimization and training strategies
- **Data-Centric ML**: Investigates how data sampling affects training dynamics
- **Adaptive Optimization**: Studies dynamic learning rate strategies that adapt during training
- **Research Methodology**: Demonstrates rigorous experimental design for ML research
- **Practical Insights**: Provides insights that can improve training efficiency and stability

**Key Research Questions**:
1. **Class-Center-Based Sampling**: Can discretizing continuous targets into pseudo-classes and using class-center samples improve or augment traditional mini-batch training?
2. **Adaptive Learning Rate as a Dynamic Parameter**: Can treating learning rate as a dynamic parameter (rather than a fixed hyperparameter) improve convergence speed and stability?
3. How do different sampling strategies affect the optimization landscape?
4. What is the relationship between data selection and convergence behavior?

## Dataset Description

**Dataset Name**: California Housing Dataset

**Source**: Scikit-learn datasets (originally from 1990 US Census, StatLib repository)

**Dataset Details**:
- **Number of samples**: 20,640 housing districts from California
- **Number of features**: 8 numerical features
  - **MedInc**: Median income in block group
  - **HouseAge**: Median house age in block group
  - **AveRooms**: Average number of rooms per household
  - **AveBedrms**: Average number of bedrooms per household
  - **Population**: Block group population
  - **AveOccup**: Average number of household members
  - **Latitude**: Block group latitude
  - **Longitude**: Block group longitude
- **Target variable**: Median house value (continuous, in hundreds of thousands of dollars)
- **Task**: Regression (predicting median house value)
- **Data quality**: Clean dataset with no missing values

**Why This Dataset**:
- **Well-understood regression problem**: Classic regression task perfect for studying training dynamics
- **Continuous target**: Ideal for exploring class-center sampling strategies on continuous targets
- **Moderate size**: ~20,640 samples is large enough to be realistic but manageable for experimentation
- **Numerical features**: All numerical features simplify preprocessing and allow focus on training strategies
- **Training dynamics**: The dataset's characteristics make it ideal for observing convergence behavior and optimization patterns

**Data Loading**:
```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X, y = housing.data, housing.target
```

**IMPORTANT**: No synthetic or hard-coded data is used in this project. All experiments use the real California Housing dataset loaded from scikit-learn's `fetch_california_housing()` function.

## Research Questions

1. Can class-center-based samples replace or augment mini-batch training?
2. Can learning rate be treated as a dynamic parameter instead of a fixed hyperparameter?
3. How do these strategies affect convergence speed, stability, and final error?

## Project Structure

```
project6_alternative_training_strategies/
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_baseline_implementation.ipynb
    ├── 02_class_center_sampling.ipynb
    ├── 03_adaptive_learning_rate.ipynb
    └── 04_comparison_and_conclusions.ipynb
```

## Key Experiments

### Baseline
- Standard Linear Regression
- Mini-batch Gradient Descent
- Fixed learning rate

### Experiment A: Class-Center-Based Sampling
- Discretize continuous target into K bins (pseudo-classes)
- Create class-center samples (mean, median, closest-to-center, synthetic)
- Compare training with:
  1. Only class-center samples
  2. Class-center + random samples
  3. Class-center + mini-batch samples

### Experiment B: Adaptive Learning Rate
- Start with high learning rate
- Monitor MSE/RMSE for spikes
- Dynamically reduce learning rate when spikes detected
- Compare against fixed learning rate and step decay

## Learning Objectives

1. Understand gradient descent variants and their trade-offs
2. Explore data-centric training strategies
3. Analyze training dynamics and convergence behavior
4. Compare alternative approaches with rigorous baselines
5. Develop research-oriented experimental methodology

## Key Insights

- Training strategies can significantly impact convergence speed
- Data sampling methods affect optimization landscape
- Adaptive learning rates can improve stability without manual tuning
- Class-center sampling provides interesting insights into dataset structure
