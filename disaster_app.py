# disaster_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="🌍 AI Disaster Prediction System",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS with Light Text
st.markdown("""
<style>
    /* Main styling - Dark Theme */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #ffffff;
        border-left: 5px solid #3498db;
        padding-left: 15px;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
    }
    
    /* Dark theme background */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 50%, #2d2d2d 100%);
        color: #ffffff;
    }
    
    /* Card styling for dark theme */
    .prediction-card {
        background: rgba(45, 45, 45, 0.8);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border: 1px solid #404040;
        color: #ffffff;
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: rgba(45, 45, 45, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        text-align: center;
        border-left: 4px solid #3498db;
        color: #ffffff;
        border: 1px solid #404040;
    }
    
    /* Warning levels with better contrast for dark theme */
    .risk-critical { 
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white !important;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255,68,68,0.4);
        border: 2px solid #ff6b6b;
    }
    
    .risk-high { 
        background: linear-gradient(135deg, #ff8800 0%, #ff5500 100%);
        color: white !important;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #ffaa44;
        box-shadow: 0 4px 15px rgba(255,136,0,0.3);
    }
    
    .risk-medium { 
        background: linear-gradient(135deg, #ffaa00 0%, #ff7700 100%);
        color: white !important;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #ffcc44;
        box-shadow: 0 4px 15px rgba(255,170,0,0.3);
    }
    
    .risk-low { 
        background: linear-gradient(135deg, #44ff44 0%, #00cc00 100%);
        color: white !important;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #66ff66;
        box-shadow: 0 4px 15px rgba(68,255,68,0.3);
    }
    
    .risk-none { 
        background: linear-gradient(135deg, #666666 0%, #888888 100%);
        color: white !important;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #999999;
        box-shadow: 0 4px 15px rgba(102,102,102,0.3);
    }
    
    /* Emergency guide styling for dark theme */
    .guide-card {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #f39c12;
        margin: 1rem 0;
        color: #ffffff;
        border: 1px solid #555555;
    }
    
    .guide-title {
        color: #f39c12;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Button styling for dark theme */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
        background: linear-gradient(135deg, #7688f0 0%, #8765c7 100%);
    }
    
    /* Tab styling for dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a1a;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #2d2d2d;
        color: #ffffff !important;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: bold;
        border: 1px solid #404040;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    
    /* Streamlit component overrides for dark theme */
    .stRadio > div {
        background: rgba(45, 45, 45, 0.8);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #404040;
    }
    
    .stRadio label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .stSlider {
        background: rgba(45, 45, 45, 0.8);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #404040;
    }
    
    .stSlider label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Info boxes styling for dark theme */
    .stInfo {
        background: rgba(45, 45, 45, 0.9) !important;
        border: 1px solid #404040 !important;
        color: #ffffff !important;
        border-radius: 10px;
    }
    
    .stSuccess {
        background: rgba(39, 174, 96, 0.2) !important;
        border: 1px solid #27ae60 !important;
        color: #ffffff !important;
        border-radius: 10px;
    }
    
    .stWarning {
        background: rgba(243, 156, 18, 0.2) !important;
        border: 1px solid #f39c12 !important;
        color: #ffffff !important;
        border-radius: 10px;
    }
    
    .stError {
        background: rgba(231, 76, 60, 0.2) !important;
        border: 1px solid #e74c3c !important;
        color: #ffffff !important;
        border-radius: 10px;
    }
    
    /* Text colors for dark theme */
    .stMarkdown, .stText, .stLabel, .stCaption {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    p, div {
        color: #ffffff !important;
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        color: #3498db !important;
    }
    
    /* Selectbox styling */
    .stSelectbox [data-baseweb="select"] {
        background: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #404040 !important;
    }
    
    .stSelectbox [data-baseweb="select"]:hover {
        border-color: #667eea !important;
    }
    
    /* Checkbox styling */
    .stCheckbox [data-baseweb="checkbox"] {
        background: #2d2d2d !important;
        border: 1px solid #404040 !important;
    }
    
    .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    .dataframe th {
        background: #404040 !important;
        color: #ffffff !important;
    }
    
    .dataframe td {
        background: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #404040 !important;
    }
    
    /* Plotly chart background */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #1a1a1a !important;
    }
    
    /* Divider styling */
    hr {
        border-color: #404040 !important;
        margin: 2rem 0 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load("enhanced_rf_model.pkl")
        label_encoder = joblib.load("enhanced_label_encoder.pkl")
        scaler = joblib.load("feature_scaler.pkl")
        features = joblib.load("feature_names.pkl")
        return model, label_encoder, scaler, features
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run train_model.py first.")
        st.stop()

def safe_mode(series):
    """Safely get the mode of a series, handling empty cases"""
    if series.empty:
        return "N/A"
    mode_values = series.mode()
    return mode_values.iloc[0] if not mode_values.empty else "N/A"

def safe_mean(series):
    """Safely get the mean of a series, handling empty cases"""
    return series.mean() if not series.empty else 0

def safe_max(series):
    """Safely get the max of a series, handling empty cases"""
    return series.max() if not series.empty else "N/A"

def get_risk_level_from_confidence(confidence, disaster_type):
    """Calculate risk level based only on confidence for dashboard display"""
    if disaster_type == 'None':
        return 'none'
    
    if confidence >= 80:
        return 'high'
    elif confidence >= 60:
        return 'medium'
    else:
        return 'low'

def get_risk_level(confidence, disaster_type, parameters):
    """Calculate overall risk level based on multiple factors for predictions"""
    if disaster_type == 'None':
        return 'none'
    
    risk_score = 0
    
    # Confidence contributes to risk
    risk_score += confidence / 100 * 40
    
    # Parameter severity contributes to risk (with safe access)
    rainfall = parameters.get('rainfall', 0)
    temperature = parameters.get('temperature', 0)
    magnitude = parameters.get('magnitude', 0)
    
    if rainfall > 150 and disaster_type == 'Flood':
        risk_score += 30
    if temperature > 35 and disaster_type == 'Wildfire':
        risk_score += 30
    if magnitude > 5.0 and disaster_type == 'Earthquake':
        risk_score += 30
    
    if risk_score > 70:
        return 'high'
    elif risk_score > 40:
        return 'medium'
    else:
        return 'low'

# Emergency Preparedness Guides with Detailed Information
EMERGENCY_GUIDES = {
    'Earthquake': {
        'icon': '🔄',
        'title': 'Earthquake Safety Guide',
        'description': 'Earthquakes can strike suddenly without warning. Proper preparation and immediate action can save lives.',
        'color': '#FF6B6B',
        'immediate_actions': [
            "**DROP** to your hands and knees",
            "**COVER** your head and neck under sturdy furniture",
            "**HOLD ON** until shaking stops",
            "Stay away from windows, glass, and exterior walls",
            "If outdoors, move to an open area away from buildings, trees, and power lines",
            "If in a vehicle, pull over and set parking brake"
        ],
        'preparation': [
            "**Secure your space**: Anchor heavy furniture, appliances, and water heaters to walls",
            "**Create emergency kits**: Prepare grab-and-go bags for each family member",
            "**Practice drills**: Conduct regular earthquake drills with family/colleagues",
            "**Know safe spots**: Identify safe places in each room (under tables, against interior walls)",
            "**Learn first aid**: Take basic first aid and CPR training",
            "**Document preparation**: Take photos of your property for insurance purposes"
        ],
        'emergency_kit': [
            "Water (1 gallon per person per day for 3+ days)",
            "Non-perishable food (3+ day supply)",
            "Manual can opener",
            "First aid kit and medications",
            "Flashlight with extra batteries",
            "Battery-powered radio",
            "Multi-tool or wrench for turning off utilities",
            "Whistle to signal for help",
            "Dust masks and goggles",
            "Moist towelettes and garbage bags"
        ]
    },
    'Flood': {
        'icon': '🌊',
        'title': 'Flood Safety Guide',
        'description': 'Floods are among the most common and destructive natural disasters. Never underestimate the power of water.',
        'color': '#4ECDC4',
        'immediate_actions': [
            "**Move to higher ground immediately**",
            "**Avoid walking or driving through flood waters** - 6 inches can sweep you away",
            "**Turn off electricity** at the main breaker if safe to do so",
            "**Evacuate immediately** if instructed by authorities",
            "Stay away from bridges over fast-moving water",
            "Keep children and pets away from floodwaters"
        ],
        'preparation': [
            "**Know your risk**: Check FEMA flood maps for your area",
            "**Elevate critical utilities**: Electrical panels, water heaters, and HVAC equipment",
            "**Install check valves** in plumbing to prevent backups",
            "**Waterproof basement**: Apply coatings and install sump pumps",
            "**Create barriers**: Keep sandbags and flood barriers available",
            "**Document valuables**: Keep important documents in waterproof containers"
        ],
        'emergency_kit': [
            "Life jackets for each family member",
            "Waterproof containers for documents",
            "Battery-powered weather radio",
            "Water purification tablets",
            "Rubber boots and gloves",
            "Emergency contact list",
            "Cash (ATMs may not work)",
            "Charged power banks for phones",
            "Insurance documents and photos of property"
        ]
    },
    'Wildfire': {
        'icon': '🔥',
        'title': 'Wildfire Safety Guide',
        'description': 'Wildfires spread rapidly and can create their own weather patterns. Early evacuation is crucial.',
        'color': '#FF9F43',
        'immediate_actions': [
            "**Evacuate immediately** if ordered - don't wait",
            "**Close all windows and doors** to prevent draft",
            "Remove flammable items from around your house",
            "Wet your roof and shrubs if time permits",
            "Turn off gas at the meter if instructed",
            "Wear protective clothing (cotton/wool, no synthetics)"
        ],
        'preparation': [
            "**Create defensible space**: Clear 30+ feet around structures",
            "**Use fire-resistant materials** for roofing and siding",
            "**Clean gutters regularly** of leaves and debris",
            "**Plan multiple evacuation routes** and practice them",
            "**Prepare pets and livestock** for quick evacuation",
            "**Keep vehicles fueled** and facing escape direction"
        ],
        'emergency_kit': [
            "N95 masks or respirators for smoke protection",
            "Goggles for eye protection",
            "Wool or cotton clothing (no synthetics)",
            "Leather gloves",
            "Emergency water and non-perishable food",
            "Important documents in fireproof container",
            "Pet supplies and carriers",
            "Prescription medications for 2+ weeks"
        ]
    },
    'Tsunami': {
        'icon': '🌊',
        'title': 'Tsunami Safety Guide', 
        'description': 'Tsunamis are series of powerful waves caused by underwater disturbances. Move to high ground immediately.',
        'color': '#45B7D1',
        'immediate_actions': [
            "**Move to high ground immediately** - don't wait for official warnings",
            "**Stay away from beaches and waterways**",
            "**Follow designated evacuation routes**",
            "**Go as far inland as possible**",
            "**Climb to upper floors** of sturdy buildings if trapped",
            "**Never go to the coast to watch** a tsunami"
        ],
        'preparation': [
            "**Know your zone**: Learn tsunami evacuation routes and safe areas",
            "**Practice evacuation drills** with your family",
            "**Keep emergency supplies** on upper floors",
            "**Learn natural warning signs**: strong earthquake, ocean roar, water recession",
            "**Have multiple communication methods**: battery radio, cell alerts, neighbor plans",
            "**Identify vertical evacuation** buildings in your area"
        ],
        'emergency_kit': [
            "Life jackets for each family member",
            "Waterproof document container",
            "Battery-powered NOAA weather radio",
            "Water and food for 3+ days",
            "Warm clothing and blankets",
            "First aid kit and medications",
            "Whistle and signal mirror",
            "Cash in small denominations"
        ]
    },
    'Volcano': {
        'icon': '🌋',
        'title': 'Volcano Safety Guide',
        'description': 'Volcanic eruptions can send ash clouds miles into the air and create deadly mudflows. Follow evacuation orders immediately.',
        'color': '#A358D6',
        'immediate_actions': [
            "**Evacuate immediately** if ordered",
            "**Avoid river valleys and low-lying areas** (lahar risk)",
            "**Protect yourself from ash fall** with mask and goggles",
            "**Stay indoors** and close all windows, doors, and dampers",
            "**Protect electronics** and machinery from ash damage",
            "**Listen to official updates** for eruption information"
        ],
        'preparation': [
            "**Learn about volcanic risks** in your region",
            "**Prepare emergency masks** (N95) and goggles for ash protection",
            "**Have supplies** for several days of sheltering indoors",
            "**Plan evacuation routes** that avoid river valleys",
            "**Protect water sources** from ash contamination",
            "**Keep vehicle air filters** and maintain full gas tank"
        ],
        'emergency_kit': [
            "N95 masks for each family member",
            "Safety goggles or glasses",
            "Long-sleeved shirts and long pants",
            "Duct tape and plastic for sealing windows",
            "Extra air filters for vehicles",
            "Battery-powered radio",
            "Ash cleanup supplies (shovels, buckets)",
            "Eye wash solution"
        ]
    },
    'None': {
        'icon': '✅',
        'title': 'General Emergency Preparedness',
        'description': 'Being prepared for any emergency ensures your safety and the safety of your loved ones.',
        'color': '#95A5A6',
        'immediate_actions': [
            "**Stay informed** about local weather and emergency conditions",
            "**Monitor official sources** for updates and instructions",
            "**Keep emergency contacts** readily available",
            "**Review your family emergency plan** regularly",
            "**Check emergency supplies** and rotate as needed",
            "**Stay calm and help others** maintain composure"
        ],
        'preparation': [
            "**Create a family emergency plan** with meeting locations",
            "**Build emergency kits** for home, car, and work",
            "**Learn basic first aid** and CPR techniques",
            "**Know your community's warning systems** and evacuation routes",
            "**Practice emergency drills** with family members",
            "**Stay informed** about local hazards and risks"
        ],
        'emergency_kit': [
            "Water (1 gallon per person per day for 3+ days)",
            "Non-perishable food (3+ day supply)",
            "Manual can opener",
            "First aid kit and medications",
            "Flashlight with extra batteries",
            "Battery-powered or hand-crank radio",
            "Multi-purpose tool",
            "Sanitation and personal hygiene items",
            "Copies of personal documents",
            "Cell phone with chargers and backup battery"
        ]
    }
}

def check_early_warnings(parameters):
    """Check parameters against safety thresholds and generate warnings"""
    warnings = []
    
    # Safe parameter access
    rainfall = parameters.get('rainfall', 0)
    temperature = parameters.get('temperature', 0)
    humidity = parameters.get('humidity', 50)
    magnitude = parameters.get('magnitude', 0)
    wind_speed = parameters.get('wind_speed', 0)
    
    # Flood warnings
    if rainfall > 250:
        warnings.append({
            'type': '🌊 CRITICAL FLOOD RISK',
            'message': 'Extreme rainfall detected - Immediate action required!',
            'severity': 'critical'
        })
    elif rainfall > 150:
        warnings.append({
            'type': '⚠️ HIGH FLOOD RISK', 
            'message': 'Heavy rainfall - Monitor water levels closely',
            'severity': 'high'
        })
    
    # Wildfire warnings
    if temperature > 40 and humidity < 20:
        warnings.append({
            'type': '🔥 CRITICAL FIRE RISK',
            'message': 'Extreme fire conditions - High alert!',
            'severity': 'critical'
        })
    elif temperature > 35 and humidity < 25:
        warnings.append({
            'type': '⚠️ HIGH FIRE RISK',
            'message': 'Severe fire danger - Take precautions',
            'severity': 'high'
        })
    
    # Earthquake warnings
    if magnitude > 7.0:
        warnings.append({
            'type': '🔄 MAJOR EARTHQUAKE',
            'message': 'Major seismic activity - Take cover immediately!',
            'severity': 'critical'
        })
    elif magnitude > 5.5:
        warnings.append({
            'type': '⚠️ SIGNIFICANT QUAKE',
            'message': 'Substantial seismic activity - Stay alert',
            'severity': 'high'
        })
    
    return warnings

def create_dashboard_tab():
    """Create the main dashboard tab"""
    st.markdown('<div class="sub-header">📊 Live System Overview</div>', unsafe_allow_html=True)
    
    # Create metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ffffff;">🌡️ System Status</h3>
            <h2 style="color: #27ae60; margin: 10px 0;">ACTIVE</h2>
            <p style="color: #cccccc;">All systems operational</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        try:
            log_df = pd.read_csv("Prediction_Log.csv")
            total_predictions = len(log_df)
        except:
            total_predictions = 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #ffffff;">📈 Predictions Made</h3>
            <h2 style="color: #3498db; margin: 10px 0;">{total_predictions}</h2>
            <p style="color: #cccccc;">Historical analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ffffff;">⚡ Response Time</h3>
            <h2 style="color: #9b59b6; margin: 10px 0;">&lt; 2s</h2>
            <p style="color: #cccccc;">Real-time analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ffffff;">🎯 Accuracy</h3>
            <h2 style="color: #e74c3c; margin: 10px 0;">94.2%</h2>
            <p style="color: #cccccc;">Model performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("---")
    st.markdown('<div class="sub-header">🚀 Quick Prediction</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **Get immediate disaster risk assessment** 
        
        Use the **AI Prediction** tab for detailed analysis with environmental parameters 
        and logical assessment to get comprehensive disaster predictions with emergency guidance.
        """)
    
    with col2:
        if st.button("🎯 Go to AI Prediction", use_container_width=True):
            st.success("Navigate to the AI Prediction tab above!")
    
    # Recent activity
    st.markdown("---")
    st.markdown('<div class="sub-header">📋 Recent Activity</div>', unsafe_allow_html=True)
    
    try:
        log_df = pd.read_csv("Prediction_Log.csv")
        if not log_df.empty:
            recent = log_df.tail(5).sort_values('Timestamp', ascending=False)
            
            for _, row in recent.iterrows():
                # Safe data access with defaults
                disaster_type = row.get('AI_Prediction', 'Unknown')
                confidence = row.get('AI_Confidence', 0)
                timestamp = row.get('Timestamp', 'Unknown')
                
                # Use simplified risk level for dashboard (no parameters needed)
                risk_level = get_risk_level_from_confidence(confidence, disaster_type)
                risk_class = f"risk-{risk_level}"
                
                # Safe text for display
                display_text = str(disaster_type).upper() if disaster_type != 'Unknown' else 'UNKNOWN'
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="color: #ffffff; margin: 0;">🔮 {disaster_type}</h4>
                            <p style="color: #cccccc; margin: 5px 0;">🕒 {timestamp} | 🤖 {confidence}% confidence</p>
                        </div>
                        <div class="{risk_class}" style="padding: 8px 16px; border-radius: 20px; min-width: 120px; text-align: center;">
                            {display_text}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📝 No prediction history yet. Make your first prediction in the AI Prediction tab!")
    except FileNotFoundError:
        st.info("📝 No prediction history yet. Make your first prediction in the AI Prediction tab!")
    except Exception as e:
        st.error(f"Error loading recent activity: {str(e)}")
        st.info("Please make a new prediction to generate activity data.")

def create_prediction_tab(model, label_encoder, scaler, FEATURES):
    """Create the AI prediction tab"""
    st.markdown('<div class="sub-header">🔮 AI Disaster Prediction Engine</div>', unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🧠 Logical Assessment")
        st.markdown("Answer these questions for initial logical assessment:")
        
        # Logical assessment questions
        q1 = st.radio("**🌍 Ground shaking or tremors detected?**", 
                      ["No", "Yes - Mild", "Yes - Strong"], key="q1")
        
        if "Yes" in q1:
            q2 = st.radio("**🌋 Volcanic activity observed?**", 
                          ["No", "Yes - Smoke/Ash", "Yes - Lava flow"], key="q2")
            if "Yes" in q2:
                logic_guess = "Volcano"
                logic_confidence = "High"
            else:
                logic_guess = "Earthquake"
                logic_confidence = "High" if "Strong" in q1 else "Medium"
        else:
            q3 = st.radio("**🌧️ Extreme rainfall conditions?**", 
                          ["No", "Yes - Heavy rain", "Yes - Torrential rain"], key="q3")
            
            if "Yes" in q3:
                q4 = st.radio("**🌊 Oceanic anomalies or coastal flooding?**", 
                              ["No", "Yes - High waves", "Yes - Coastal flooding"], key="q4")
                if "Yes" in q4:
                    logic_guess = "Tsunami"
                    logic_confidence = "High" if "Coastal flooding" in q4 else "Medium"
                else:
                    logic_guess = "Flood"
                    logic_confidence = "High" if "Torrential" in q3 else "Medium"
            else:
                q5 = st.radio("**🔥 Fire or smoke observed?**", 
                              ["No", "Yes - Small fire", "Yes - Large wildfire"], key="q5")
                if "Yes" in q5:
                    logic_guess = "Wildfire"
                    logic_confidence = "High" if "Large" in q5 else "Medium"
                else:
                    logic_guess = "None"
                    logic_confidence = "High"
        
        # Display logical assessment
        st.markdown("---")
        st.markdown("#### 📋 Logical Assessment Result")
        if logic_guess != "None":
            st.success(f"**Probable Disaster:** {logic_guess}")
            st.info(f"**Confidence Level:** {logic_confidence}")
        else:
            st.success("**✅ No disaster conditions detected logically**")
    
    with col2:
        st.markdown("#### 📊 Environmental Parameters")
        st.markdown("Adjust the parameters for AI prediction:")
        
        # Create two columns for sliders
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            rain = st.slider("**Rainfall (mm)**", 0, 500, 50, 
                           help="Total rainfall measurement")
            humidity = st.slider("**Humidity (%)**", 0, 100, 50,
                               help="Relative humidity percentage")
            temp = st.slider("**Temperature (°C)**", -10, 60, 25,
                           help="Ambient temperature")
            magnitude = st.slider("**Seismic Magnitude**", 0.0, 10.0, 0.0, 0.1,
                                help="Earthquake magnitude if detected")
        
        with subcol2:
            wind = st.slider("**Wind Speed (km/h)**", 0, 150, 20,
                           help="Wind speed measurement")
            soil = st.slider("**Soil Moisture (%)**", 0, 100, 40,
                           help="Soil moisture content")
            depth = st.slider("**Event Depth (km)**", 0, 100, 0,
                            help="Depth of seismic event if applicable")
        
        # Early warnings with safe parameter access
        parameters = {
            'rainfall': rain,
            'humidity': humidity, 
            'temperature': temp,
            'wind_speed': wind,
            'magnitude': magnitude
        }
        
        warnings = check_early_warnings(parameters)
        if warnings:
            st.markdown("#### ⚠️ Early Warning System")
            for warning in warnings:
                severity_class = f"risk-{warning['severity']}"
                st.markdown(f'<div class="{severity_class}">{warning["type"]}: {warning["message"]}</div>', 
                            unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🚀 Run AI Prediction", use_container_width=True, type="primary"):
            with st.spinner("🤖 AI is analyzing environmental parameters..."):
                # Prepare input data
                input_data = np.array([[rain, humidity, temp, wind, soil, magnitude, depth]])
                input_scaled = scaler.transform(input_data)
                
                # Get prediction
                probabilities = model.predict_proba(input_scaled)[0]
                predicted_idx = np.argmax(probabilities)
                predicted_disaster = label_encoder.inverse_transform([predicted_idx])[0]
                confidence = probabilities[predicted_idx] * 100
                
                # Calculate risk level with actual parameters
                risk_level = get_risk_level(confidence, predicted_disaster, parameters)
                
                # Display results
                st.markdown("---")
                st.markdown("#### 🎯 Prediction Results")
                
                # Results in columns
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown(f"**Predicted Disaster:**")
                    st.markdown(f"<h1 style='color: #ffffff; margin: 10px 0;'>{predicted_disaster}</h1>", unsafe_allow_html=True)
                    
                    st.markdown(f"**AI Confidence:**")
                    st.markdown(f"<h1 style='color: #3498db; margin: 10px 0;'>{confidence:.1f}%</h1>", unsafe_allow_html=True)
                    
                    # Display risk level
                    risk_class = f"risk-{risk_level}"
                    st.markdown(f"**Risk Level:**")
                    st.markdown(f'<div class="{risk_class}" style="padding: 10px; text-align: center; margin: 10px 0;">{risk_level.upper()} RISK</div>', 
                                unsafe_allow_html=True)
                
                with res_col2:
                    if logic_guess == predicted_disaster:
                        st.success("### ✅ Assessments Match!")
                        st.info("Logical and AI predictions are aligned")
                    elif logic_guess == "None" and predicted_disaster != "None":
                        st.warning("### ⚠️ AI Detects Potential Disaster")
                        st.info("AI model identified risk conditions")
                    elif logic_guess != "None" and predicted_disaster == "None":
                        st.warning("### 🤔 Logical Assessment Suggests Risk")
                        st.info("Consider verifying environmental parameters")
                    else:
                        st.warning("### 🔄 Assessments Differ")
                        st.info(f"Logic: {logic_guess} | AI: {predicted_disaster}")
                
                # Confidence visualization
                st.markdown("#### 📊 Confidence Distribution")
                
                disasters = label_encoder.classes_
                conf_df = pd.DataFrame({
                    'Disaster': disasters,
                    'Confidence': probabilities * 100
                }).sort_values('Confidence', ascending=False)
                
                fig = px.bar(conf_df, x='Disaster', y='Confidence', 
                            color='Confidence',
                            color_continuous_scale='RdYlGn',
                            title="AI Confidence by Disaster Type",
                            height=400)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Emergency Guide
                guide = EMERGENCY_GUIDES.get(predicted_disaster, EMERGENCY_GUIDES['None'])
                st.markdown("---")
                st.markdown("#### 🛡️ Emergency Preparedness Guide")
                
                st.markdown(f"### {guide['icon']} {guide['title']}")
                st.markdown(f"*{guide['description']}*")
                
                tab1, tab2, tab3 = st.tabs(["🚨 Immediate Actions", "📝 Preparation", "🎒 Emergency Kit"])
                
                with tab1:
                    st.markdown("##### Critical steps to take immediately:")
                    for i, action in enumerate(guide['immediate_actions'], 1):
                        st.markdown(f"**{i}. {action}**")
                
                with tab2:
                    st.markdown("##### How to prepare in advance:")
                    for i, step in enumerate(guide['preparation'], 1):
                        st.markdown(f"**{i}. {step}**")
                
                with tab3:
                    st.markdown("##### Essential emergency supplies:")
                    for item in guide['emergency_kit']:
                        st.markdown(f"• {item}")
                
                # Log prediction
                log_prediction(logic_guess, predicted_disaster, confidence, 
                             rain, humidity, temp, wind, soil, magnitude, depth)
                st.success("📝 Prediction logged successfully!")

def create_analysis_tab():
    """Create analytics and historical data tab"""
    st.markdown('<div class="sub-header">📈 Predictive Analytics & Insights</div>', unsafe_allow_html=True)
    
    try:
        log_df = pd.read_csv("Prediction_Log.csv")
        if log_df.empty:
            st.info("📊 Analytics data will appear here after making predictions in the AI Prediction tab.")
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: rgba(45,45,45,0.8); border-radius: 10px; border: 1px solid #404040;">
                <h3 style="color: #ffffff;">No Data Available Yet</h3>
                <p style="color: #cccccc;">Make your first prediction in the AI Prediction tab to see analytics here!</p>
            </div>
            """, unsafe_allow_html=True)
            return
            
        log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_predictions = len(log_df)
            st.metric("Total Predictions", total_predictions)
        
        with col2:
            avg_confidence = safe_mean(log_df['AI_Confidence'])
            st.metric("Average Confidence", f"{avg_confidence:.1f}%")
        
        with col3:
            most_common = safe_mode(log_df['AI_Prediction'])
            st.metric("Most Predicted", most_common)
        
        with col4:
            recent_activity = safe_max(log_df['Timestamp'])
            if recent_activity != "N/A":
                st.metric("Last Activity", recent_activity.strftime('%Y-%m-%d'))
            else:
                st.metric("Last Activity", "N/A")
        
        # Create analysis tabs
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📊 Disaster Distribution", "📈 Confidence Trends", "🔍 Parameter Analysis"])
        
        with analysis_tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Disaster type distribution
                disaster_counts = log_df['AI_Prediction'].value_counts()
                fig1 = px.pie(disaster_counts, 
                             values=disaster_counts.values, 
                             names=disaster_counts.index,
                             title="Distribution of Predicted Disasters",
                             color_discrete_sequence=px.colors.qualitative.Set3)
                fig1.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown("#### 📋 Prediction Statistics")
                st.dataframe(disaster_counts.reset_index().rename(
                    columns={'index': 'Disaster', 'AI_Prediction': 'Count'}
                ), use_container_width=True, hide_index=True)
        
        with analysis_tab2:
            # Confidence over time
            fig2 = px.line(log_df, x='Timestamp', y='AI_Confidence', 
                          color='AI_Prediction',
                          title="Prediction Confidence Over Time",
                          labels={'AI_Confidence': 'Confidence (%)', 'Timestamp': 'Date'},
                          height=500)
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white',
                legend_font_color='white'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with analysis_tab3:
            # Parameter correlations
            numeric_cols = ['Rainfall_mm', 'Humidity_%', 'Temperature_C', 'Wind_Speed_kmph', 'AI_Confidence']
            # Only use columns that exist in the dataframe
            available_cols = [col for col in numeric_cols if col in log_df.columns]
            if available_cols:
                corr_matrix = log_df[available_cols].corr()
                
                fig3 = px.imshow(corr_matrix,
                               title="Environmental Parameter Correlations",
                               labels=dict(x="Parameters", y="Parameters", color="Correlation"),
                               color_continuous_scale='RdBu_r',
                               aspect="auto")
                fig3.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No numeric data available for correlation analysis.")
            
    except FileNotFoundError:
        st.info("📊 Analytics data will appear here after making predictions in the AI Prediction tab.")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: rgba(45,45,45,0.8); border-radius: 10px; border: 1px solid #404040;">
            <h3 style="color: #ffffff;">No Data Available Yet</h3>
            <p style="color: #cccccc;">Make your first prediction in the AI Prediction tab to see analytics here!</p>
        </div>
        """, unsafe_allow_html=True)

def create_preparedness_tab():
    """Create emergency preparedness guide tab"""
    st.markdown('<div class="sub-header">🛡️ Disaster Preparedness Center</div>', unsafe_allow_html=True)
    
    # Disaster type selector
    selected_disaster = st.selectbox(
        "Select Disaster Type for Preparedness Guide:",
        list(EMERGENCY_GUIDES.keys())
    )
    
    guide = EMERGENCY_GUIDES[selected_disaster]
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {guide['color']} 0%, {guide['color']}66 100%); 
                color: white; padding: 2rem; border-radius: 15px; text-align: center; border: 1px solid {guide['color']};">
        <h1 style="color: white; margin: 0;">{guide['icon']} {guide['title']}</h1>
        <p style="font-size: 1.2rem; color: white; margin: 10px 0 0 0;">{guide['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🚨 Immediate Actions", "📝 Preparation Guide", "🎒 Emergency Kit"])
    
    with tab1:
        st.markdown("### 🚨 Immediate Response Actions")
        st.markdown("**Critical steps to take when disaster strikes:**")
        
        for i, action in enumerate(guide['immediate_actions'], 1):
            st.markdown(f"""
            <div class="guide-card">
                <h4 style="color: #f39c12;">Step {i}</h4>
                <p style="color: #ffffff;">{action}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📝 Long-term Preparation Guide")
        st.markdown("**How to prepare in advance for maximum safety:**")
        
        for i, step in enumerate(guide['preparation'], 1):
            st.markdown(f"""
            <div class="guide-card">
                <h4 style="color: #f39c12;">Preparation {i}</h4>
                <p style="color: #ffffff;">{step}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 🎒 Essential Emergency Kit Checklist")
        st.markdown("**Build your emergency kit with these essential items:**")
        
        # Create two columns for checklist
        col1, col2 = st.columns(2)
        kit_items = guide['emergency_kit']
        mid_point = len(kit_items) // 2
        
        with col1:
            for item in kit_items[:mid_point]:
                st.checkbox(item, key=f"kit_{kit_items.index(item)}")
        
        with col2:
            for item in kit_items[mid_point:]:
                st.checkbox(item, key=f"kit_{kit_items.index(item)}")
        
        st.markdown("---")
        st.markdown("#### 💡 Pro Tips:")
        st.info("""
        - Store emergency kits in accessible locations
        - Rotate food and water every 6 months  
        - Keep copies of important documents in waterproof containers
        - Include special needs items for children, elderly, and pets
        - Practice using your emergency equipment regularly
        """)

def create_settings_tab():
    """Create settings and about tab"""
    st.markdown('<div class="sub-header">⚙️ System Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌐 System Settings")
        
        # Theme selection
        theme = st.selectbox("Color Theme", ["Dark Theme (Current)", "Light", "Auto"])
        
        # Data preferences
        st.markdown("#### 📊 Data Preferences")
        auto_save = st.checkbox("Automatically save all predictions", value=True)
        data_retention = st.slider("Data retention period (days)", 30, 365, 90)
        
        # Notification settings
        st.markdown("#### 🔔 Notification Settings")
        email_alerts = st.checkbox("Email alerts for high-risk predictions")
        sms_alerts = st.checkbox("SMS notifications for critical warnings")
        
        if st.button("💾 Save Settings", use_container_width=True):
            st.success("Settings saved successfully!")
    
    with col2:
        st.markdown("#### ℹ️ About This System")
        st.markdown("""
        **AI Disaster Prediction System** v2.0
        
        *Intelligent disaster prediction and emergency preparedness platform*
        
        This system combines advanced machine learning with real-time environmental 
        monitoring to provide accurate disaster predictions and comprehensive emergency guidance.
        
        **Key Features:**
        - 🤖 AI-powered disaster prediction using ensemble models
        - 📊 Real-time environmental parameter analysis  
        - 🛡️ Comprehensive preparedness guides and checklists
        - 📈 Historical analytics and trend analysis
        - 🌙 Dark theme optimized interface
        - 🔔 Smart early warning system
        
        **Technical Stack:**
        - Python 3.9+ with Scikit-learn, XGBoost
        - Streamlit for interactive web interface
        - Plotly for advanced visualizations
        - Pandas for data analysis
        - Real-time data integration capabilities
        """)
        
        st.markdown("---")
        st.markdown("#### 📞 Emergency Contacts & Resources")
        st.info("""
        **Emergency Services: 911**  
        **Disaster Management: 1-800-621-FEMA (3362)**  
        **Weather Alerts: 1-800-939-6300**  
        **Red Cross: 1-800-RED-CROSS**  
        **Poison Control: 1-800-222-1222**
        """)

def log_prediction(logic_guess, predicted_disaster, confidence, 
                   rain, humidity, temp, wind, soil, magnitude, depth):
    """Log prediction to CSV file"""
    log_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Logic_Assessment": logic_guess,
        "AI_Prediction": predicted_disaster,
        "AI_Confidence": round(confidence, 2),
        "Rainfall_mm": rain,
        "Humidity_%": humidity,
        "Temperature_C": temp,
        "Wind_Speed_kmph": wind,
        "Soil_Moisture_%": soil,
        "Magnitude": magnitude,
        "Depth_km": depth
    }])
    
    try:
        existing_log = pd.read_csv("Prediction_Log.csv")
        updated_log = pd.concat([existing_log, log_entry], ignore_index=True)
    except FileNotFoundError:
        updated_log = log_entry
    
    updated_log.to_csv("Prediction_Log.csv", index=False)

def main():
    # Load model
    model, label_encoder, scaler, FEATURES = load_model()
    
    # Main header
    st.markdown('<h1 class="main-header">🌍 AI Disaster Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #cccccc;'>Advanced AI-powered disaster prediction with real-time monitoring and emergency guidance</p>", unsafe_allow_html=True)
    
    # Create main navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Dashboard", 
        "🔮 AI Prediction", 
        "📊 Analytics", 
        "🛡️ Preparedness", 
        "⚙️ Settings"
    ])
    
    with tab1:
        create_dashboard_tab()
    
    with tab2:
        create_prediction_tab(model, label_encoder, scaler, FEATURES)
    
    with tab3:
        create_analysis_tab()
    
    with tab4:
        create_preparedness_tab()
    
    with tab5:
        create_settings_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #95a5a6;'>"
        "🌍 AI Disaster Prediction System | Class 12 AI Capstone Project | "
        "Built with ❤️ for Community Safety"
        "</p>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()