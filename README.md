# Airflow + MLflow + Grafana: End-to-End Pipeline

## 📋 Overview

This project implements an automated ML workflow that:
- **Trains** an IsolationForest model for anomaly detection on traffic accident data
- **Tracks** experiments, parameters, metrics, and models using MLflow
- **Orchestrates** the entire pipeline with Apache Airflow
- **Monitors** experiment results and metrics in Grafana dashboards

## 🏗️ Architecture

The system consists of four containerized services working together:

```
┌─────────────┐      ┌──────────────┐      ┌────────────┐
│   Airflow   │─────▶│    MLflow    │─────▶│ PostgreSQL │
│ (Scheduler) │      │   (Tracking) │      │  (Storage) │
└─────────────┘      └──────────────┘      └────────────┘
       │                     │
       │                     ▼
       │              ┌──────────────┐
       └─────────────▶│   Grafana    │
                      │ (Monitoring) │
                      └──────────────┘
```

### Component Interaction Flow:
1. **Airflow** triggers DAG tasks on schedule or manually
2. DAG tasks execute Python scripts (train/score)
3. Scripts log to **MLflow** tracking server via HTTP API
4. MLflow stores metadata in **PostgreSQL** database
5. MLflow stores artifacts (models, files) using artifact proxy
6. **Grafana** queries PostgreSQL to visualize experiment metrics

## 🔧 Tools

### Apache Airflow
**Purpose**: Workflow orchestration and task scheduling

**How it works**:
- Runs in standalone mode with LocalExecutor for task execution
- Defines workflows as DAGs (Directed Acyclic Graphs) in Python
- Executes tasks sequentially with dependency management
- Provides web UI for monitoring and triggering runs
- Automatically retries failed tasks based on configuration

**In this project**:
- Orchestrates the `traffic_anomaly_detection_mlflow` DAG
- Task 1: `train_model` - Installs dependencies and trains IsolationForest
- Task 2: `score_model` - Scores data with latest model and generates visualizations
- Connects to MLflow via `http://mlflow:5000` using Docker networking
- Stores DAG definitions and scripts in `/opt/airflow/dags`

**Configuration**:
- Executor: LocalExecutor (sequential task execution)
- Database: SQLite for Airflow metadata
- Scheduler: Manual trigger only (schedule=None)

---

### MLflow
**Purpose**: Machine learning lifecycle management and experiment tracking

**How it works**:
- **Tracking Server**: REST API for logging parameters, metrics, and artifacts
- **Backend Store**: PostgreSQL database stores run metadata, parameters, metrics
- **Artifact Store**: File storage for models, plots, and data files
- **Artifact Proxy**: MLflow server proxies artifact uploads/downloads via HTTP
- **Model Registry**: (Not used in this setup, but available for model versioning)

**In this project**:
- Tracking server listens on port 5000
- Logs training parameters (n_estimators, contamination, use_date_features)
- Records metrics (anomaly_rate, training time)
- Saves trained model as artifact using scikit-learn flavor
- Provides web UI to browse experiments, compare runs, view artifacts
- Uses `--serve-artifacts` flag to handle artifact uploads via HTTP instead of shared filesystem (this solves the permission denied errors encountered when using the direct filesystem access)

**Configuration**:
- Backend store URI: `postgresql://mlflow:mlflow123@postgres:5432/mlflow_db`
- Artifact destination: `/mlflow/artifacts` (in mlflow-server container)
- Host validation: Accepts requests from `mlflow`, `localhost`, `127.0.0.1`

---

### PostgreSQL
**Purpose**: Persistent storage for MLflow tracking data

**How it works**:
- Relational database stores structured experiment data
- Tables include: experiments, runs, metrics, params, tags, artifacts
- Provides ACID guarantees for data consistency
- Queried by both MLflow (for tracking) and Grafana (for visualization)

**In this project**:
- Database name: `mlflow_db`
- Stores all MLflow experiment metadata
- Accessed by MLflow tracking server for writes
- Accessed by Grafana datasource for reads
- Persistent volume ensures data survives container restarts

**Configuration**:
- User: `mlflow`
- Password: `mlflow123`
- Port: 5432
- Volume: `postgres_data` for persistence

---

### Grafana
**Purpose**: Data visualization and monitoring dashboards

**How it works**:
- Connects to data sources (PostgreSQL in this case)
- Executes SQL queries to retrieve metrics
- Renders data as time series charts, tables, stats panels
- Auto-refreshes dashboards at configurable intervals
- Supports provisioning for automated setup

**In this project**:
- Pre-configured PostgreSQL datasource pointing to `mlflow_db`
- Pre-loaded dashboard: "MLflow Anomaly Detection Metrics"
- Displays: total runs, latest anomaly rate, trend charts, run history
- Visualizes data from MLflow's `metrics`, `experiments`, and `runs` tables
- Also mounts visualization PNG files from Airflow for reference

**Configuration**:
- Admin user: `admin` / `admin123`
- Datasource: PostgreSQL on `postgres:5432`
- Dashboard provisioning: `/etc/grafana/provisioning/dashboards`
- Data provisioning: `/etc/grafana/provisioning/datasources`

## 📊 Machine Learning Pipeline

### Use Case: Traffic Accident Anomaly Detection

**Goal**: Identify unusual patterns in traffic accident data that may indicate special events, data quality issues, or systemic changes.

**Model**: IsolationForest (scikit-learn)
- Unsupervised anomaly detection algorithm
- Works by isolating outliers through random partitioning
- Assigns anomaly scores (-1 for anomalies, 1 for normal)

**Workflow**:

#### 1. Training Phase (`train_iforest.py`)
```
Load Data → Preprocess → Train IsolationForest → Log to MLflow → Save Model
```

**Steps**:
- Reads `traffic_accidents.csv`
- Preprocesses: drops unnecessary columns, handles missing values
- Optional: Extracts date features (year, month, day, hour, weekday)
- Trains IsolationForest with 200 estimators, 0.1 contamination
- Logs parameters: `n_estimators`, `contamination`, `use_date_features`
- Logs metrics: `anomaly_rate` (percentage of anomalies detected)
- Saves model using MLflow's scikit-learn flavor
- MLflow creates experiment "traffic_anomaly_detection" if not exists

#### 2. Scoring Phase (`score_iforest.py`)
```
Load Latest Model → Score Data → Generate Visualizations → Save Results
```

**Steps**:
- Queries MLflow API to find latest run in experiment
- Loads trained model using `mlflow.sklearn.load_model()`
- Scores data: predicts anomaly labels and calculates decision function scores
- Generates 5 visualizations:
  1. **Anomaly Score Distribution**: Histogram showing score spread
  2. **Anomaly Percentage**: Pie chart of normal vs anomaly ratio
  3. **Top Anomalies**: Bar chart of 10 highest anomaly scores
  4. **Anomaly Trend**: Time series (if date features available)
  5. **Summary Statistics**: Text file with counts and percentages
- Saves scored results to CSV with predictions and scores

## 🚀 Quick Start

See [README_SETUP.md](README_SETUP.md) for detailed setup instructions.

**TL;DR**:
```powershell
# Start services
docker-compose up -d

# Get Airflow password
docker logs airflow-standalone 2>&1 | Select-String "password"

# Access services
# Airflow: http://localhost:8080
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000
```

## 📁 Project Structure

```
.
├── docker-compose.yml              # Multi-container orchestration
├── README.md                       # This file
├── README_SETUP.md                 # Detailed setup guide
│
├── airflow/                        # Airflow workspace
│   ├── dags/
│   │   ├── dag_anomaly_mlflow.py   # DAG definition
│   │   ├── train_iforest.py        # Training script
│   │   ├── score_iforest.py        # Scoring script
│   │   ├── requirements.txt        # Python dependencies
│   │   ├── traffic_accidents.csv   # Sample dataset
│   │   └── visualizations/         # Generated charts
│   ├── logs/                       # Task execution logs
│   └── plugins/                    # Custom Airflow plugins
│
└── grafana/                        # Grafana configuration
    ├── provisioning/
    │   ├── datasources/            # Auto-config PostgreSQL
    │   └── dashboards/             # Auto-load dashboards
    └── dashboards/
        └── mlflow_anomaly_dashboard.json  # Pre-built dashboard
```

## 🔄 Data Flow

1. **User triggers DAG** in Airflow UI
2. **train_model task** executes:
   - Installs Python packages from requirements.txt
   - Runs train_iforest.py with dataset
   - Script logs to MLflow tracking server
   - MLflow stores metadata in PostgreSQL
   - MLflow stores model artifact via artifact proxy
3. **score_model task** executes:
   - Queries MLflow for latest run ID
   - Downloads model from MLflow artifact store
   - Scores new/same data
   - Generates visualization PNG files
   - Saves results to CSV
4. **Grafana displays** updated metrics from PostgreSQL

## 🎯 Key Features

✅ **Containerized**: All services run in Docker for portability  
✅ **Persistent**: Data survives container restarts via volumes  
✅ **Automated**: DAG handles end-to-end workflow orchestration  
✅ **Tracked**: All experiments logged with parameters and metrics  
✅ **Monitored**: Real-time dashboard shows experiment history  
✅ **Reproducible**: Models versioned and retrievable by run ID  
✅ **Scalable**: Can extend to multiple workers, remote artifact storage  

## 🛠️ Customization

### Add More Metrics
Edit `train_iforest.py`:
```python
mlflow.log_metric("accuracy", accuracy_score)
mlflow.log_metric("precision", precision_score)
```

### Change Model
Replace IsolationForest with any scikit-learn model:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
```

### Schedule DAG
Edit `dag_anomaly_mlflow.py`:
```python
schedule="0 0 * * *"  # Daily at midnight
```

### Add More Visualizations
Edit `score_iforest.py` and add to `create_visualizations()` function.

## 📈 Monitoring & Observability

**Airflow UI** (`http://localhost:8080`):
- View DAG structure and dependencies
- Monitor task execution status
- Check task logs for debugging
- Trigger manual runs
- View execution history

**MLflow UI** (`http://localhost:5000`):
- Browse all experiments and runs
- Compare runs side-by-side
- Download model artifacts
- View parameter and metric history
- Search and filter runs

**Grafana Dashboards** (`http://localhost:3000`):
- Real-time metrics visualization
- Anomaly rate trends over time
- Experiment run statistics
- Quick overview of latest results

## 🔐 Security Notes

⚠️ **This is a development setup** - not production-ready:
- No authentication on MLflow
- Default Grafana password
- PostgreSQL credentials in docker-compose.yml
- Containers run with default users

For production:
- Enable MLflow authentication
- Use secrets management for credentials
- Run containers as non-root users
- Use HTTPS with SSL certificates
- Implement network policies


