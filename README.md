# ⚽ Unlocking the Game: Predictive Power and Player Potential with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/)

> An In-Depth Machine Learning Journey Through 17,000+ Footballers — Exploring Performance, Value, and Future Stars Using SoFIFA's Comprehensive Dataset. This project achieves **97.37% accuracy** in player position classification and **95.87% R² score** in market value prediction.

---

## 📋 Table of Contents

- [Features](#-features)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Results](#-results)
- [Environment Variables](#-environment-variables)
- [Resources](#-resources)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **Multi-Model Machine Learning Pipeline**: Implements and compares multiple algorithms
  - **Regression Models**: Linear Regression for value and rating prediction
  - **Classification Models**: 
    - Logistic Regression (92.15% accuracy)
    - Random Forest Classifier (97.37% accuracy)
    - Support Vector Machine (92.26% accuracy)

- **Comprehensive Data Analysis**: 
  - Exploratory Data Analysis (EDA) with 20+ visualizations
  - Player market value prediction (R² = 0.9587)
  - Overall rating prediction (R² = 0.9171)
  - Position classification (97.37% accuracy)

- **Advanced Feature Engineering**: 
  - Date parsing and temporal features
  - Position hierarchy extraction
  - Name tokenization
  - Categorical encoding

- **Rich Visualizations**: 
  - Player distribution by nationality, age, and position
  - Skill correlation heatmaps
  - Market value analysis
  - Rating distribution and player comparisons

- **Data Preprocessing**: 
  - Missing value imputation
  - Label encoding for categorical features
  - Feature scaling with StandardScaler
  - Train/test split with stratification

---

## 🎯 Model Performance

### Regression Models

#### Market Value Prediction (value_euro)
| Metric | Value |
|--------|-------|
| **R² Score** | **95.87%** |
| **Mean Squared Error** | 1,370,878,656,677.66 |
| **Model Type** | Linear Regression |

#### Overall Rating Prediction (overall_rating)
| Metric | Value |
|--------|-------|
| **R² Score** | **91.71%** |
| **Mean Squared Error** | 3.95 |
| **Model Type** | Linear Regression |

### Classification Models (Position Prediction)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **97.37%** | **98%** | **97%** | **98%** |
| **SVM** | 92.26% | 94% | 93% | 93% |
| **Logistic Regression** | 92.15% | 94% | 93% | 93% |

**Best Model**: Random Forest Classifier with **97.37% accuracy** 🏆

### Position-wise Classification Performance (Random Forest)

| Position | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Defender | 96% | 98% | 97% | 1,333 |
| Midfielder | 97% | 99% | 98% | 1,149 |
| Goalkeeper | 100% | 100% | 100% | 370 |
| Striker | 100% | 93% | 96% | 688 |

---

## 📊 Dataset

### Source
The project uses the FIFA Players Dataset from Kaggle:

- **Football Players Data** by Maso0dahmed
  - [Dataset Link](https://www.kaggle.com/datasets/maso0dahmed/football-players-data)

### Dataset Overview
```
FIFA Players Dataset
├── Total Players: 17,954
├── Total Features: 51 columns
└── Data Source: SoFIFA
```

### Key Features

#### Player Information
- **Personal**: Name, Full Name, Birth Date, Age, Nationality
- **Physical**: Height (cm), Weight (kg), Body Type, Preferred Foot
- **Professional**: Positions, Club, Overall Rating, Potential

#### Financial Data
- **Value Euro**: Player market value (10K - 110.5M)
- **Wage Euro**: Weekly wages
- **Release Clause Euro**: Contract release amount

#### Performance Attributes (30+ Skills)
- **Attacking**: Finishing, Volleys, Heading Accuracy, Shot Power, Long Shots
- **Technical**: Dribbling, Ball Control, Crossing, Curve, Freekick Accuracy
- **Physical**: Acceleration, Sprint Speed, Stamina, Strength, Jumping, Agility
- **Mental**: Vision, Composure, Reactions, Positioning
- **Defensive**: Marking, Interceptions, Standing Tackle, Sliding Tackle

### Data Preprocessing Steps

1. **Missing Value Handling**:
   - Numerical columns: Median imputation
   - Categorical columns: "nothing" placeholder
   - Dropped columns with >50% missing values (national_team features)

2. **Feature Engineering**:
   - Extracted year, month, day from birth_date
   - Split positions into main and sub-positions
   - Tokenized full names into first, second, last names
   - Created position hierarchy: GK → Goalkeeper, RB/LB/CB → Defender, etc.

3. **Encoding**:
   - Label Encoding for 160+ nationalities
   - Position encoding (4 main categories)
   - Body type and preferred foot encoding

4. **Feature Scaling**:
   - StandardScaler for numerical features
   - Applied to all ML models

### Class Distribution

| Position Category | Count | Percentage |
|-------------------|-------|------------|
| Defender | 6,665 | 37.1% |
| Midfielder | 5,745 | 32.0% |
| Striker | 3,440 | 19.2% |
| Goalkeeper | 1,850 | 10.3% |

---

## 📁 Project Structure

```
FOOTBALL/
│
├── result/
│   └── exploratory data analysis/
│       ├── Age_Distribution.png
│       ├── aggression_penalties.png
│       ├── correlation_of_attributes.png
│       ├── finishing_and_composure.png
│       ├── freekick_accuracy_and_curve.png
│       ├── H,W_high-rated_Players.png
│       ├── nationality_distribution.png
│       ├── Numerical_col_distribution_violin.png
│       ├── Numerical_col_distribution.png
│       ├── overall_rating_potential.png
│       ├── overall_rating.png
│       ├── players_count_position.png
│       ├── Players_Rating.png
│       ├── rating_distribution.png
│       ├── Skills_Correlation.png
│       ├── top_frequent_country.png
│       ├── value_by_rating.png
│       ├── vision_and_positioning.png
│       └── weight_and_agility.png
│
├── result/
│   ├── overall_rating_regression.txt
│   ├── positions_classification_model_1.txt    # Logistic Regression
│   ├── positions_classification_model_2.txt    # Random Forest
│   ├── positions_classification_model_3.txt    # SVM
│   └── value_euro_regression.txt
│
├── src/
│   └── football.ipynb                          # Main analysis notebook
│
├── dataset.txt                                 # Dataset download link
├── kaggle_notebook.txt                         # Link to Kaggle notebook
├── requirements.txt                            # Python dependencies
├── .env                                        # Environment variables
├── .env.example                                # Environment variables template
├── .gitignore                                  # Git ignore rules
├── LICENSE                                     # Project license
└── README.md                                   # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (recommended)
- 5GB+ free disk space

### Installation Steps

#### 1. Install Python using MiniConda

Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)

Create a new environment:
```bash
conda create -n Football_Analysis python=3.8
```

Activate the environment:
```bash
conda activate Football_Analysis
```

#### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/football-players-analysis.git
cd football-players-analysis
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Download Dataset

**Option A: Using Kaggle API**
```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('maso0dahmed/football-players-data')"
```

**Option B: Manual Download**
- Download dataset from the link in `dataset.txt`
- Place `fifa_players.csv` in the `/kaggle/input/football-players-data/` directory

#### 5. Setup Environment Variables

```bash
cp .env.example .env
```

Edit `.env` file:
```env
PROJECT_VERSION=1.0
TEST_SIZE=0.2
RANDOM_STATE=42
```

---

## 💻 Usage

### Running the Analysis

Open and run the Jupyter notebook:
```bash
jupyter notebook src/football.ipynb
```

Or run as a Python script:
```bash
python src/football.py
```

### Predicting Player Market Value

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

# Load trained model
model = pickle.load(open('value_euro_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Prepare new player data
player_data = {
    'age': 25,
    'height_cm': 180,
    'weight_kgs': 75,
    'overall_rating': 85,
    'potential': 88,
    'preferred_foot': 1,  # Encoded: 0=Left, 1=Right
    'positions': 1,  # Encoded: 0=Defender, 1=Midfielder, 2=GK, 3=Striker
    # ... add all required features
}

# Convert to DataFrame
X_new = pd.DataFrame([player_data])

# Scale features
X_new_scaled = scaler.transform(X_new)

# Predict value
predicted_value = model.predict(X_new_scaled)
print(f"Predicted Market Value: €{predicted_value[0]:,.2f}")
```

### Predicting Player Position

```python
from sklearn.ensemble import RandomForestClassifier

# Load trained model
rf_model = pickle.load(open('position_classifier_rf.pkl', 'rb'))

# Prepare player attributes
player_features = {
    'pace': 85,
    'shooting': 75,
    'passing': 80,
    'dribbling': 82,
    'defending': 45,
    'physical': 70,
    # ... add all required features
}

# Convert to DataFrame and scale
X_player = pd.DataFrame([player_features])
X_player_scaled = scaler.transform(X_player)

# Predict position
position = rf_model.predict(X_player_scaled)
positions_map = {0: 'Defender', 1: 'Midfielder', 2: 'Goalkeeper', 3: 'Striker'}
print(f"Predicted Position: {positions_map[position[0]]}")

# Get probability distribution
probabilities = rf_model.predict_proba(X_player_scaled)[0]
for i, prob in enumerate(probabilities):
    print(f"{positions_map[i]}: {prob*100:.2f}%")
```

### Exploratory Data Analysis

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('fifa_players.csv')

# Top 10 most valuable players
top_players = data.nlargest(10, 'value_euro')[['full_name', 'value_euro', 'overall_rating']]
print(top_players)

# Average rating by nationality (top 10 countries)
avg_rating_by_country = data.groupby('nationality')['overall_rating'].mean().nlargest(10)
plt.figure(figsize=(12, 6))
avg_rating_by_country.plot(kind='bar')
plt.title('Top 10 Countries by Average Player Rating')
plt.show()

# Skill correlation heatmap
skill_columns = ['crossing', 'finishing', 'dribbling', 'ball_control', 'acceleration', 'stamina']
plt.figure(figsize=(10, 8))
sns.heatmap(data[skill_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Skill Attributes Correlation')
plt.show()
```

---

## 🔧 Technical Details

### Machine Learning Pipeline

#### 1. Data Preprocessing
```python
# Missing value imputation
data['release_clause_euro'].fillna(data['release_clause_euro'].median(), inplace=True)

# Drop high-null columns
data.drop(['national_team', 'national_rating', 'national_team_position', 
           'national_jersey_number'], axis=1, inplace=True)

# Remove remaining nulls
data.dropna(inplace=True)
```

#### 2. Feature Engineering
```python
# Date features
data['year'] = data['birth_date'].dt.year
data['month'] = data['birth_date'].dt.month
data['day'] = data['birth_date'].dt.day

# Position hierarchy
data['main_positions'] = data['positions'].str.split(',').str[0]
data['sub_positions'] = data['positions'].str.split(',').str[1]

# Position mapping
mapping = {
    'GK': 'goalkeeper',
    'RB': 'defender', 'LB': 'defender', 'CB': 'defender',
    'CAM': 'Midfielder', 'CM': 'Midfielder', 'CDM': 'Midfielder',
    'CF': 'striker', 'LW': 'striker', 'RW': 'striker', 'ST': 'striker'
}
data['positions'] = data['main_positions'].map(mapping)
```

#### 3. Encoding and Scaling
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Label encoding
le = LabelEncoder()
categorical_columns = ['nationality', 'positions', 'body_type', 'preferred_foot']
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Model Architectures

#### Linear Regression (Value Prediction)
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):,.2f}")
```

#### Random Forest Classifier (Position Classification)
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Test Size | 20% |
| Random State | 42 |
| Scaling Method | StandardScaler |
| Cross Validation | None (single split) |
| Feature Selection | Manual (dropped names) |

### Model Comparison

| Task | Best Model | Metric | Score |
|------|------------|--------|-------|
| Value Prediction | Linear Regression | R² | 95.87% |
| Rating Prediction | Linear Regression | R² | 91.71% |
| Position Classification | Random Forest | Accuracy | 97.37% |

---

## 📈 Results

### Exploratory Data Analysis Insights

1. **Nationality Distribution**:
   - England has the most players (1,662)
   - Top 10 countries represent 45% of all players
   - 160 different nationalities in dataset

2. **Market Value Distribution**:
   - Range: €10,000 - €110,500,000
   - Median value: €750,000
   - Most players (35%) valued between €100K - €1M

3. **Age Distribution**:
   - Average age: 25.1 years
   - Peak player count: 21-23 years
   - Range: 16-47 years

4. **Position Distribution**:
   - Defenders: 37.1%
   - Midfielders: 32.0%
   - Strikers: 19.2%
   - Goalkeepers: 10.3%

5. **Overall Rating**:
   - Average: 66.2
   - High-rated players (85+): 4.2%
   - Strong correlation with market value (r=0.82)

### Key Correlations

**With Market Value**:
- Overall Rating: 0.82
- Potential: 0.75
- Reactions: 0.68
- Composure: 0.65

**Skill Correlations**:
- Ball Control ↔ Dribbling: 0.91
- Short Passing ↔ Vision: 0.87
- Standing Tackle ↔ Marking: 0.89

### Model Insights

1. **Value Prediction (R² = 0.9587)**:
   - Most important features: Overall rating, potential, reputation
   - Model explains 95.87% of variance in player values
   - Performs well across all value ranges

2. **Rating Prediction (R² = 0.9171)**:
   - Technical skills (ball control, dribbling) most predictive
   - Age has moderate negative correlation with peak performance
   - Position-specific attributes show varied importance

3. **Position Classification (97.37% accuracy)**:
   - Goalkeepers classified with 100% accuracy (distinct attributes)
   - Defenders vs. Midfielders: Most challenging distinction
   - Physical attributes key for position determination

---

## 🌐 Environment Variables

Create a `.env` file with the following variables:

```env
# Project Configuration
PROJECT_NAME=Football_Players_Analysis
PROJECT_VERSION=1.0

# Data Paths
DATA_PATH=/kaggle/input/football-players-data/fifa_players.csv
OUTPUT_PATH=/kaggle/working/

# Model Parameters
TEST_SIZE=0.2
RANDOM_STATE=42
CV_FOLDS=5

# Feature Engineering
MIN_POSITION_FREQUENCY=100
VALUE_BINS=10

# Model Configuration
# Regression
LINEAR_REG_FIT_INTERCEPT=True

# Classification
RF_N_ESTIMATORS=100
RF_MAX_DEPTH=None
RF_MIN_SAMPLES_SPLIT=2
RF_MIN_SAMPLES_LEAF=1

LR_MAX_ITER=1000
LR_SOLVER=lbfgs

SVM_KERNEL=rbf
SVM_C=1.0
SVM_GAMMA=scale

# Visualization
FIGURE_SIZE_SMALL=(10, 6)
FIGURE_SIZE_MEDIUM=(12, 8)
FIGURE_SIZE_LARGE=(15, 10)
```

---

## 📚 Resources

### Model Results Files
All model results and predictions available in `/result/`:
- `value_euro_regression.txt` - Market value prediction results
- `overall_rating_regression.txt` - Player rating prediction results
- `positions_classification_model_1.txt` - Logistic Regression results
- `positions_classification_model_2.txt` - Random Forest results
- `positions_classification_model_3.txt` - SVM results

### Kaggle Notebook
Complete interactive notebook with all visualizations:
- See `kaggle_notebook.txt` for the link

### Dataset
Download link available in `dataset.txt`:
- Football Players Data (Maso0dahmed)

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [FIFA Ratings Explained](https://www.ea.com/games/fifa/fifa-22/ratings)
- [SoFIFA Database](https://sofifa.com/)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide for Python code
- Add docstrings to functions and classes
- Update README.md for significant changes
- Test code before submitting PR
- Include visualizations for new analyses
- Document new features and model improvements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Abdelhady Ali** - *Initial work* - [MyGitHub](https://github.com/Abdelhady-22)

---

## 🙏 Acknowledgments

- **Dataset**: Thanks to Maso0dahmed for providing the comprehensive FIFA Players dataset
- **SoFIFA**: For maintaining the most detailed football player database
- **Kaggle**: For providing the platform and community support
- **Scikit-learn**: For powerful machine learning algorithms
- **EA Sports FIFA**: For inspiring this analysis
- **Football Analytics Community**: For insights and methodologies
- **Open Source Community**: For various tools and libraries

---

## 📞 Contact

For questions, suggestions, or collaborations:

- **Email**: abdelhady2322005@gmail.com
- **LinkedIn**: [My LinkedIn](https://www.linkedin.com/in/abdelhady-ali-940761316)
- **GitHub**: [My GitHub](https://github.com/Abdelhady-22)
- **Kaggle**: [My Kaggle Profile](https://www.kaggle.com/abdulhadialimohamed)
---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The models and predictions should not be used as the sole basis for:
- Player recruitment decisions
- Transfer negotiations
- Contract valuations
- Professional scouting evaluations

**Important Notes:**
- Model predictions are based on FIFA game ratings, not real-world performance
- Market values are estimates and may not reflect actual transfer fees
- Player potential is speculative and based on game mechanics
- Real football performance involves factors beyond statistical analysis
- Always consult professional scouts and analysts for real-world decisions

---

## 📊 Citations

### Dataset Citation
```bibtex
@dataset{maso0dahmed_fifa_2024,
  title={Football Players Data},
  author={Maso0dahmed},
  year={2024},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/maso0dahmed/football-players-data}
}
```

---

## 🔮 Future Improvements

- [ ] Implement time-series analysis for player value trends
- [ ] Add transfer prediction model
- [ ] Create player comparison tool
- [ ] Build injury risk prediction model
- [ ] Implement deep learning for attribute prediction
- [ ] Add web scraping for real-time updates
- [ ] Create interactive dashboard with Plotly/Dash
- [ ] Develop player recommendation system
- [ ] Add team composition optimization
- [ ] Implement career trajectory prediction
- [ ] Create mobile app for player scouting
- [ ] Add multi-language support
- [ ] Integrate with football APIs
- [ ] Build player performance tracking system

---

## 📖 Research Applications

This project can be used for:
- **Sports Analytics**: Understanding player valuation factors
- **Machine Learning Education**: End-to-end ML pipeline example
- **Data Science Portfolio**: Demonstrating EDA and modeling skills
- **Transfer Market Analysis**: Market value prediction research
- **Player Development**: Identifying potential growth patterns
- **Scouting Automation**: Assisting in player discovery
- **Academic Research**: Football analytics studies

---

<div align="center">

**Made with ⚽ for advancing football analytics and data science in sports**

⭐ Star this repo if you find it helpful!

[![GitHub stars](https://img.shields.io/github/stars/Abdelhady-22/Football?style=social)](https://github.com/Abdelhady-22/Football)
[![GitHub forks](https://img.shields.io/github/forks/Abdelhady-22/Football?style=social)](https://github.com/Abdelhady-22/Football)
[![GitHub watchers](https://img.shields.io/github/watchers/Abdelhady-22/Football?style=social)](https://github.com/Abdelhady-22/Football)

</div>