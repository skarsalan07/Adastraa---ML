# 🌟 AdAstraa AI (24h ML + Django Challenge)

**Sale Amount Prediction — ML + Django Full-Stack Application**

A complete end-to-end solution built in 24 hours, including:
- ✅ Cleaning + preprocessing messy real-world marketing data
- ✅ Training a regression model to predict `Sale_Amount`
- ✅ Hosting the trained model inside a Django backend
- ✅ Uploading `test.csv` (without `Sale_Amount`)
- ✅ Running preprocessing + prediction
- ✅ Downloading `predictions.csv` with `Predicted_Sale_Amount`
- ✅ Supporting interactive & ML-focused visualizations

---

## 📁 Project Structure

```bash
adastraa-ml-challenge/
│── config/                  # Django project config
│── prediction/              # Django app (views, forms, templates)
│── ml/
│   ├── preprocess.py        # Data cleaning & preprocessing pipeline
│   ├── train_model.py       # Training script
│   └── pipeline.pkl         # Saved trained model
│── data/
│   └── train.csv            # Provided messy dataset
│── staticfiles/             # For Render deployment
│── templates/               # App templates (upload + visuals)
│── manage.py
│── Procfile
│── requirements.txt
│── README.md

```
# 🎯 1. Data Cleaning & Preprocessing
## The dataset contained a mix of real-world issues:

| Issue                             | Handling                              |
| --------------------------------- | ------------------------------------- |
| Inconsistent casing               | Lowercased + stripped                 |
| Typos in keywords, location names | Normalized via lowercasing            |
| Cost values with `$` or commas    | Cleaned and converted to numeric      |
| Mixed date formats                | Multi-format parsing + fallback       |
| Missing numeric values            | Median imputation                     |
| Incorrect/missing Conversion Rate | Recomputed using Conversions / Clicks |
| Duplicate rows                    | Removed                               |
| Lack of feature columns           | Extracted: Year, Month, DayOfWeek     |
| Outliers                          | Handled via RandomForest robustness   |


## Final Model-Ready Features : Clicks ,Impressions ,Cost ,Leads ,Conversions, Conversion_Rate ,Ad_Year ,Ad_Month, Ad_DayOfWeek, Campaign_Name ,Location ,Device ,Keyword

# 🤖 2. Modeling Approach
## Algorithm Used: RandomForestRegressor

### Why Random Forest
- Excellent for tabular business data.
- Handles noisy & messy inputs.
- Robust to outliers.
- No need for feature scaling.
- Works well with OneHotEncoded categorical variables.
- Produces useful feature importance.

## [Raw Data] -> [Preprocessor] -> [RandomForestRegressor] -> [Predictions]

## Model Workflow : 
- Load messy dataset
- Apply custom preprocessing
- Fit preprocessing + model pipeline
- Validate model
- Save trained model → ml/pipeline.pkl

# 3. Django Web Application
- ✅ Upload test.csv (without Sale_Amount)
- ✅ Apply same preprocessing as during training
- ✅ Predict Sale_Amount
- ✅ Generate downloadable predictions.csv

# 📊 4. Visualizations
The application includes:
- ✅ Feature Importance 
- ✅ Sale_Amount Distribution
## ✅ Input Feature Visualizer:
- Scatter Plot
- Histogram
- Box Plot
- KDE Plot

# 🚀 5. Run Application Locally
1️⃣ Clone the Repository : 

git clone https://github.com/skarsalan07/Adastraa---ML.git



2️⃣ Install Dependencies

- cd Adastraa---ML
- pip install -r requirements.txt

3️⃣ Run Django Server

- python manage.py runserver

4️⃣ Visit
- 👉 http://127.0.0.1:8000/

# 📝 6. Assumptions & Limitations
## ✅ Assumptions

- Test CSV matches training CSV schema
-- Only Sale_Amount is missing in test.csv
--Unseen categories handled via handle_unknown="ignore" in OHE
  
# 🚀 7. Future Improvements
- Add database logs for uploaded files
- Add user login/authentication
- Convert frontend to React + Tailwind
- Deploy using Docker
- Add CI/CD pipeline
- Add monitoring dashboards
- MLOPS and DVC integration for continue monitoring and retraining

🏗 8. Scaling to Production
A production-grade architecture could include:
- S3 for file upload storage
- FastAPI microservice for ML inference
- Redis queue + Celery for async jobs
- PostgreSQL for logs & user management
- API gateway + load balancer


