# train_model.py
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class DisasterDataCollector:
    def __init__(self):
        self.df = pd.DataFrame()
    
    def fetch_earthquake_data(self):
        """Fetch real earthquake data from USGS"""
        try:
            url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
            params = {
                'format': 'geojson',
                'starttime': '2020-01-01',
                'endtime': '2024-01-01',
                'minmagnitude': 4.5,
                'limit': 1000
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            earthquake_data = []
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                # Simulate environmental features based on earthquake characteristics
                earthquake_data.append({
                    'Rainfall_mm': np.random.uniform(0, 50),
                    'Humidity_%': np.random.uniform(30, 80),
                    'Temperature_C': np.random.uniform(10, 35),
                    'Wind_Speed_kmph': np.random.uniform(5, 40),
                    'Soil_Moisture_%': np.random.uniform(20, 60),
                    'Magnitude': props['mag'],
                    'Depth_km': coords[2],
                    'Disaster_Type': 'Earthquake',
                    'Confidence_Score': min(0.9, props['mag'] / 10 + 0.3)
                })
            
            return pd.DataFrame(earthquake_data)
        except Exception as e:
            print(f"Error fetching earthquake data: {e}")
            return pd.DataFrame()
    
    def fetch_weather_disaster_data(self):
        """Generate realistic weather-related disaster data"""
        disasters = []
        
        # Flood data (high rainfall scenarios)
        for _ in range(200):
            disasters.append({
                'Rainfall_mm': np.random.uniform(150, 400),
                'Humidity_%': np.random.uniform(75, 100),
                'Temperature_C': np.random.uniform(15, 30),
                'Wind_Speed_kmph': np.random.uniform(20, 60),
                'Soil_Moisture_%': np.random.uniform(80, 100),
                'Magnitude': np.random.uniform(3, 8),
                'Depth_km': 0,
                'Disaster_Type': 'Flood',
                'Confidence_Score': np.random.uniform(0.7, 0.95)
            })
        
        # Wildfire data (hot, dry conditions)
        for _ in range(150):
            disasters.append({
                'Rainfall_mm': np.random.uniform(0, 10),
                'Humidity_%': np.random.uniform(10, 30),
                'Temperature_C': np.random.uniform(30, 50),
                'Wind_Speed_kmph': np.random.uniform(25, 70),
                'Soil_Moisture_%': np.random.uniform(0, 20),
                'Magnitude': np.random.uniform(2, 6),
                'Depth_km': 0,
                'Disaster_Type': 'Wildfire',
                'Confidence_Score': np.random.uniform(0.7, 0.95)
            })
        
        # Tsunami data (coastal conditions)
        for _ in range(100):
            disasters.append({
                'Rainfall_mm': np.random.uniform(50, 150),
                'Humidity_%': np.random.uniform(60, 90),
                'Temperature_C': np.random.uniform(20, 35),
                'Wind_Speed_kmph': np.random.uniform(40, 100),
                'Soil_Moisture_%': np.random.uniform(50, 80),
                'Magnitude': np.random.uniform(4, 9),
                'Depth_km': np.random.uniform(0, 5),
                'Disaster_Type': 'Tsunami',
                'Confidence_Score': np.random.uniform(0.7, 0.95)
            })
        
        # Volcano data (unique conditions)
        for _ in range(80):
            disasters.append({
                'Rainfall_mm': np.random.uniform(0, 50),
                'Humidity_%': np.random.uniform(40, 80),
                'Temperature_C': np.random.uniform(25, 45),
                'Wind_Speed_kmph': np.random.uniform(10, 50),
                'Soil_Moisture_%': np.random.uniform(10, 40),
                'Magnitude': np.random.uniform(3, 7),
                'Depth_km': np.random.uniform(0, 10),
                'Disaster_Type': 'Volcano',
                'Confidence_Score': np.random.uniform(0.7, 0.95)
            })
        
        return pd.DataFrame(disasters)
    
    def add_normal_conditions(self):
        """Add normal weather conditions for balanced dataset"""
        normal_conditions = []
        for _ in range(500):
            normal_conditions.append({
                'Rainfall_mm': np.random.uniform(0, 100),
                'Humidity_%': np.random.uniform(30, 70),
                'Temperature_C': np.random.uniform(10, 30),
                'Wind_Speed_kmph': np.random.uniform(5, 30),
                'Soil_Moisture_%': np.random.uniform(20, 60),
                'Magnitude': 0,
                'Depth_km': 0,
                'Disaster_Type': 'None',
                'Confidence_Score': np.random.uniform(0.1, 0.3)
            })
        return pd.DataFrame(normal_conditions)
    
    def build_dataset(self):
        """Build comprehensive dataset"""
        print("🔄 Building disaster dataset...")
        
        # Fetch real earthquake data
        eq_df = self.fetch_earthquake_data()
        
        # Generate other disaster data
        weather_df = self.fetch_weather_disaster_data()
        
        # Add normal conditions
        normal_df = self.add_normal_conditions()
        
        # Combine all data
        self.df = pd.concat([eq_df, weather_df, normal_df], ignore_index=True)
        
        # Remove any potential duplicates or invalid rows
        self.df = self.df.dropna()
        self.df = self.df[self.df['Confidence_Score'] > 0]
        
        print(f"✅ Dataset built with {len(self.df)} samples")
        print(f"📊 Class distribution:\n{self.df['Disaster_Type'].value_counts()}")
        
        return self.df

def train_model():
    """Train the enhanced disaster prediction model"""
    
    # Build dataset
    collector = DisasterDataCollector()
    df = collector.build_dataset()
    
    # Prepare features and target
    FEATURES = ['Rainfall_mm', 'Humidity_%', 'Temperature_C', 'Wind_Speed_kmph', 
                'Soil_Moisture_%', 'Magnitude', 'Depth_km']
    
    X = df[FEATURES]
    y = df['Disaster_Type']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with balanced class weights
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Model Accuracy: {accuracy:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save artifacts
    joblib.dump(clf, "enhanced_rf_model.pkl")
    joblib.dump(le, "enhanced_label_encoder.pkl")
    joblib.dump(scaler, "feature_scaler.pkl")
    joblib.dump(FEATURES, "feature_names.pkl")
    
    # Save the dataset for reference
    df.to_csv("enhanced_disaster_dataset.csv", index=False)
    
    print("✅ Enhanced model and artifacts saved successfully!")
    print(f"📁 Files saved: enhanced_rf_model.pkl, enhanced_label_encoder.pkl, feature_scaler.pkl")
    
    return clf, le, scaler, FEATURES

if __name__ == "__main__":
    train_model()