import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json
import os
import pickle

def generate_mock_claims_data():
    """Simulates realistic healthcare records to avoid data privacy restrictions."""
    np.random.seed(42)
    n_patients = 2500
    
    data = {
        'patient_id': range(10000, 10000 + n_patients),
        'Age': np.random.randint(45, 90, n_patients),
        'Charlson_Comorbidity': np.random.poisson(3, n_patients),
        'Num_Medications': np.random.randint(2, 25, n_patients),
        'Prior_Admissions': np.random.poisson(1, n_patients),
        'Dual_Eligible': np.random.choice([0, 1], n_patients, p=[0.75, 0.25]) # 1 = Low income
    }
    df = pd.DataFrame(data)
    

    log_odds = (0.02 * df['Age'] + 0.18 * df['Charlson_Comorbidity'] + 
                0.30 * df['Prior_Admissions'] - 0.45 * df['Dual_Eligible'] - 2.8)
    probabilities = 1 / (1 + np.exp(-log_odds))
    df['readmitted_30d'] = np.random.binomial(1, probabilities)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_claims.csv", index=False)
    return df

def execute_governance_pipeline():
    print("[1/3] Generating operational data vectors...")
    df = generate_mock_claims_data()
        
    features = ['Age', 'Charlson_Comorbidity', 'Num_Medications', 'Prior_Admissions', 'Dual_Eligible']
    X = df[features]
    y = df['readmitted_30d']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[2/3] Training clinical predictive models...")
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    with open("models/readmission_model.pkl", "wb") as f:
        pickle.dump(model, f)
    X_train.to_csv("data/X_train_sample.csv", index=False)
    
    print("[3/3] Executing structural 4/5ths Rule Bias Audit...")
    df_audit = X_test.copy()
    df_audit['pred_y'] = model.predict(X_test)
    
    rates = df_audit.groupby('Dual_Eligible')['pred_y'].mean()
    disparate_impact_ratio = rates.get(1, 0.0) / rates.get(0, 1.0) if rates.get(0, 1.0) > 0 else 1.0
    
    audit_results = {
        "disparate_impact_ratio": round(disparate_impact_ratio, 3),
        "audit_disposition": "FAIL (Systemic Bias Found)" if disparate_impact_ratio < 0.8 else "PASS"
    }
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/audit_results.json", "w") as f:
        json.dump(audit_results, f, indent=4)
        
    print("\n=========================================")
    print("      PIPELINE EXECUTION COMPLETE        ")
    print(f"  Disparate Impact Ratio: {audit_results['disparate_impact_ratio']}")
    print(f"  Audit Disposition:      {audit_results['audit_disposition']}")
    print("=========================================\n")

if __name__ == '__main__':
    execute_governance_pipeline()
