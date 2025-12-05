# Ray Parallel Model Training Notebooks

## 1. `ray_cpu_model_training.ipynb` - CPU Cluster Notebook

**Location**: `notebooks/ray_cpu_model_training.ipynb`

**Purpose**: Trains 90 traditional ML models on CPU cluster

- **Models**: Logistic Regression (15), SVM (10), Random Forest (20), XGBoost (20), Gradient Boosting (15), Naive Bayes (10)
- **Cluster**: 8 workers, 32 cores per node (256 total cores)
- **Resource Allocation**:
  - Single-core models: 1 CPU (Logistic Regression, SVM, Naive Bayes)
  - Multi-core models: 4 CPUs (Random Forest, XGBoost, Gradient Boosting)
- **Model IDs**: 0-89
- **Output**: Appends to `ryuta.ray.model_training_results`

## 2. `ray_gpu_model_training.ipynb` - GPU Cluster Notebook

**Location**: `notebooks/ray_gpu_model_training.ipynb`

**Purpose**: Trains 10 PyTorch deep learning models on GPU cluster

- **Models**: PyTorch MLP (10 models with various architectures)
- **Cluster**: Single-node with 4 GPUs
- **Resource Allocation**: 1 GPU per model
- **Model IDs**: 90-99
- **Output**: Appends to `ryuta.ray.model_training_results`

## 3. `ray_model_analysis.ipynb` - Analysis Notebook

**Location**: `notebooks/ray_model_analysis.ipynb`

**Purpose**: Comprehensive analysis of all 100 models

- **Reads from** shared Delta table: `ryuta.ray.model_training_results`
- **Analyses includes**:
  - Top 10 best performing models
  - Performance by model type
  - CPU vs GPU comparison
  - Feature selection strategy analysis
  - Training time and efficiency metrics
  - Visualizations (box plots, scatter plots, bar charts)
  - Key insights and recommendations
- **Summary table**: `ryuta.ray.model_training_summary`

## 🔄 Execution Workflow

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│   CPU Cluster (256 cores)   │     │   GPU Cluster (4 GPUs)      │
│                             │     │                             │
│  ray_cpu_model_training     │     │  ray_gpu_model_training     │
│  ↓                          │     │  ↓                          │
│  90 Traditional ML Models   │     │  10 PyTorch MLP Models      │
│  (IDs: 0-89)                │     │  (IDs: 90-99)               │
└──────────────┬──────────────┘     └──────────────┬──────────────┘
               │                                    │
               └────────────┬───────────────────────┘
                            ↓
              ryuta.ray.model_training_results
                            ↓
              ┌─────────────────────────────┐
              │   ray_model_analysis        │
              │   ↓                         │
              │   Comprehensive Analysis    │
              │   + Visualizations          │
              └─────────────────────────────┘
```

## ✨ Key Features

1. **Independent Execution**: Both training notebooks can run simultaneously on their respective clusters
2. **Shared Results Table**: Both write to the same Delta table with append mode
3. **No ID Conflicts**: CPU models (0-89), GPU models (90-99)
4. **Comprehensive Analysis**: Third notebook provides unified analysis of all results
5. **Bayesian Optimization**: Each model uses Optuna for hyperparameter tuning
6. **Feature Diversity**: 7 different feature selection strategies
7. **Progress Tracking**: Real-time progress updates during training


## 🚀 How to Use

1. **Run CPU notebook** on your CPU cluster (8 workers, 32 cores each)
2. **Run GPU notebook** on your GPU cluster (single-node, 4 GPUs) - can be run in parallel with step 1
3. **Run analysis notebook** after both complete to see comprehensive results


## 📊 Expected Results

After running both training notebooks, you will have:
- 90 traditional ML models trained and evaluated
- 10 deep learning models trained and evaluated
- All results stored in a unified Delta table
- Comprehensive analysis with visualizations
- Insights on best performing models and strategies

## 🔧 Prerequisites

- Databricks workspace with access to CPU and GPU clusters
- Ray installed on both clusters
- Delta Lake enabled
- Synthetic dataset table: `ryuta.ray.synthetic_data`

## 📦 Dependencies

- Ray
- PyTorch
- scikit-learn
- XGBoost
- Optuna
- pandas
- numpy
- matplotlib
- seaborn
- PySpark

## 💡 Tips

- Monitor cluster resource utilization during training
- CPU notebook can handle 60+ concurrent jobs with 256 cores
- GPU notebook trains 4 models in parallel (one per GPU)
- Total expected runtime: 10-30 minutes depending on cluster performance
- Results are automatically saved after each model completes (fault-tolerant)
