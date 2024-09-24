# train_model.py

# Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from ydata_profiling import ProfileReport
import shap
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

def load_data():
    """
    Load the California Housing dataset.
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    print("\nData loaded successfully.")
    return df

def feature_engineering(df):
    """
    Perform feature engineering by adding interaction terms.
    """
    df['Rooms_per_House'] = df['AveRooms'] / df['HouseAge']
    print("Feature engineering complete: Added 'Rooms_per_House'.")
    return df

def scaling_normalization(df):
    """
    Scale the features using StandardScaler.
    """
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print("Scaling and normalization complete.")
    
    # Define features (X) and target (y)
    X = df_scaled.drop('MedHouseVal', axis=1)
    y = df_scaled['MedHouseVal']
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Data splitting into training and testing sets complete.")
    
    return X_train, X_test, y_train, y_test, X

def generate_eda_report(df):
    """
    Generate an automated EDA report using Pandas Profiling.
    """
    profile = ProfileReport(df, title="California Housing Profiling Report", explorative=True)
    profile.to_file("california_housing_profile_report.html")
    print("Pandas Profiling report generated and saved as 'california_housing_profile_report.html'.")

def model_training(X_train, y_train):
    """
    Train the RandomForestRegressor model with hyperparameter tuning using GridSearchCV.
    """
    model = RandomForestRegressor()
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    
    # Fit GridSearchCV
    print("\nStarting hyperparameter tuning with GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print("Best Model Parameters:", grid_search.best_params_)
    
    return best_model

def model_evaluation(best_model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    """
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nModel Evaluation:\nMean Squared Error (MSE): {mse}\nR² Score: {r2}")
    return mse, r2

def explain_model(best_model, X_test, X_train):
    """
    Provide model interpretability using SHAP and LIME.
    """
    # SHAP
    print("\nGenerating SHAP summary plot...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
    plt.savefig('shap_summary_plot.png')
    plt.close()
    print("SHAP summary plot saved as 'shap_summary_plot.png'.")
    
    # LIME
    print("Generating LIME explanation for a single instance...")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        mode='regression'
    )
    
    # Select an instance to explain (e.g., first instance in test set)
    instance = X_test.iloc[0].values
    exp = lime_explainer.explain_instance(instance, best_model.predict, num_features=5)
    exp.save_to_file('lime_explanation.html')
    print("LIME explanation saved as 'lime_explanation.html'.")

def save_model(best_model):
    """
    Save the trained model to a file.
    """
    joblib.dump(best_model, 'california_housing_model.pkl')
    print("Trained model saved as 'california_housing_model.pkl'.")

def main():
    # Task 1: Data Collection and Preprocessing
    df = load_data()  # 1.1 Load Data
    df = feature_engineering(df)  # 1.2 Feature Engineering
    X_train, X_test, y_train, y_test, X = scaling_normalization(df)  # 1.3 Scaling/Normalization
    generate_eda_report(df)  # 1.4 AutoEDA with Pandas Profiling
    
    # Task 2: Model Selection, Training, and Hyperparameter Tuning
    best_model = model_training(X_train, y_train)  # 2.1 & 2.2 Model Training and Hyperparameter Tuning
    mse, r2 = model_evaluation(best_model, X_test, y_test)  # 2.3 Model Evaluation
    
    # Task 3: Explainable AI (XAI) Implementation
    explain_model(best_model, X_test, X_train)  # 3.1 Apply SHAP and LIME
    
    # Save the trained model
    save_model(best_model)

if __name__ == "__main__":
    main()
