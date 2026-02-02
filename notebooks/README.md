# Ray Parallel Model Training and Inference Notebooks

## 1. `ray_cpu_model_training_ray_tune.ipynb` - CPU Cluster Training Notebook

**Location**: `notebooks/ray_cpu_model_training_ray_tune.ipynb`

**Purpose**: Trains 90 traditional ML models on CPU cluster using Ray Tune for distributed HPO

- **Models**: Logistic Regression (15), LinearSVC (10), Random Forest (20), XGBoost (20), LightGBM (15), Naive Bayes (10)
- **Cluster**: 8 workers, 32 cores per node (256 total cores)
- **Resource Allocation**:
  - Single-core models: 1 CPU (Logistic Regression, SVM, Naive Bayes)
  - Multi-core models: 4 CPUs (Random Forest, XGBoost, LightGBM)
- **Model IDs**: 0-89
- **Output**: Models registered to Unity Catalog as `ryuta.ray.cpu_model_*_ray_tune`

## 2. `ray_gpu_model_training.ipynb` - GPU Cluster Training Notebook

**Location**: `notebooks/ray_gpu_model_training.ipynb`

**Purpose**: Trains 10 PyTorch deep learning models on GPU cluster

- **Models**: PyTorch MLP (10 models with various architectures)
- **Cluster**: Multi-node with 4+ GPUs (g4dn.xlarge recommended)
- **Resource Allocation**: 1 GPU per model
- **Model IDs**: 90-99
- **Output**: Models registered to Unity Catalog as `ryuta.ray.gpu_model_*_child`

## 3. `ray_cpu_batch_inference.ipynb` - Distributed Batch Inference Notebook

**Location**: `notebooks/ray_cpu_batch_inference.ipynb`

**Purpose**: Performs distributed batch inference using Ray Core with registered CPU models

- **Loads**: Latest versions of `cpu_model_*_child` models from Unity Catalog
- **Cluster**: CPU cluster (8 workers, 256 total cores)
- **Features**:
  - Parallel model loading and inference via Ray Core
  - All model predictions written to a single Delta table
  - Per-model accuracy analysis (if labels available)
- **Output Table**: `ryuta.ray.batch_inference_results` - Predictions from all models
- **MLflow Tracking**: Logs inference run metrics and model list

## 4. `generate_synthetic_data.ipynb` - Data Generation Notebook

**Location**: `notebooks/generate_synthetic_data.ipynb`

**Purpose**: Generates synthetic classification dataset for training and inference

- **Features**: 100 features (50 informative, 25 redundant, 25 noise)
- **Samples**: 10,000 rows
- **Labels**: Binary classification (60/40 class balance)
- **Output**: `ryuta.ray.synthetic_data`

## 🔄 Execution Workflow

```
┌──────────────────────────────────┐     ┌─────────────────────────────────┐
│   CPU Cluster (256 cores)        │     │   GPU Cluster (4+ GPUs)         │
│                                  │     │                                 │
│  ray_cpu_model_training_ray_tune │     │  ray_gpu_model_training         │
│  ↓                               │     │  ↓                              │
│  90 Traditional ML Models        │     │  10 PyTorch MLP Models          │
│  (IDs: 0-89)                     │     │  (IDs: 90-99)                   │
│  ↓                               │     │  ↓                              │
│  Register to Unity Catalog       │     │  Register to Unity Catalog      │
│  ryuta.ray.cpu_model_*_ray_tune  │     │  ryuta.ray.gpu_model_*_child    │
└──────────────┬───────────────────┘     └──────────────┬──────────────────┘
               │                                        │
               └────────────────┬───────────────────────┘
                                ↓
                   Unity Catalog Model Registry
                                ↓
               ┌────────────────────────────────────┐
               │   ray_cpu_batch_inference          │
               │   (CPU Cluster - 256 cores)        │
               │   ↓                                │
               │   Distributed Batch Inference      │
               │   • Load cpu_model_*_child models  │
               │   • Parallel inference via Ray     │
               │   • Ensemble predictions           │
               └──────────────┬─────────────────────┘
                              ↓
               ┌────────────────────────────────────┐
               │   Output Delta Tables              │
               │   • batch_inference_results        │
               │   • model_predictions              │
               └────────────────────────────────────┘
```

## ✨ Key Features

1. **Independent Execution**: Both training notebooks can run simultaneously on their respective clusters
2. **Unity Catalog Model Registry**: All models registered to Unity Catalog for governance and versioning
3. **No ID Conflicts**: CPU models (0-89), GPU models (90-99)
4. **Distributed Batch Inference**: Ray Core enables parallel model loading and inference
5. **Ensemble Predictions**: Aggregates predictions from all models by averaging probabilities
6. **Ray Tune HPO**: CPU models use Ray Tune for distributed hyperparameter optimization
7. **MLflow Tracking**: Parent-child run structure for organized experiment tracking
8. **Feature Diversity**: 7 different feature selection strategies
9. **Progress Tracking**: Real-time progress updates during training and inference


## 🚀 How to Use

1. **Generate data** using `generate_synthetic_data.ipynb`
2. **Run CPU training notebook** on your CPU cluster (8 workers, 32 cores each)
3. **Run GPU training notebook** on your GPU cluster (multi-node, 4+ GPUs) - can run in parallel with step 2
4. **Run batch inference notebook** after training completes to generate predictions


## 📊 Expected Results

After running the full workflow, you will have:
- 90 traditional ML models trained and registered to Unity Catalog
- 10 deep learning models trained and registered to Unity Catalog
- Ensemble batch inference results in Delta table
- Per-model predictions for detailed analysis
- MLflow tracking for all training and inference runs

## 🔧 Prerequisites

- Databricks workspace with access to CPU and GPU clusters
- Ray installed on both clusters
- Delta Lake enabled
- Unity Catalog configured with appropriate catalog and schema
- Synthetic dataset table: `ryuta.ray.synthetic_data`

## 📦 Dependencies

- Ray (with ray.util.spark for Databricks integration)
- Ray Tune (for distributed HPO)
- PyTorch
- scikit-learn
- XGBoost
- LightGBM
- Optuna
- MLflow
- pandas
- numpy
- PySpark

## 💡 Tips

- Monitor cluster resource utilization during training
- CPU notebook can handle 60+ concurrent jobs with 256 cores
- GPU notebook trains 4 models in parallel (one per GPU)
- Total expected training runtime: 10-30 minutes depending on cluster performance
- Batch inference is highly parallelized - all models run inference concurrently
- Models are registered to Unity Catalog for easy versioning and governance
- Use MLflow UI to explore parent-child run relationships

## 📁 Output Tables

| Table | Description |
|-------|-------------|
| `ryuta.ray.synthetic_data` | Source data for training and inference |
| `ryuta.ray.batch_inference_results` | Predictions from all models (row_index, model_name, probability, prediction) |
