import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hashlib
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from duckduckgo_search import DDGS

try:
    from ultralytics import YOLO
except ImportError:
    st.error("ultralytics package not installed.")

st.set_page_config(page_title="Fuel Economy Dashboard", layout="wide")

IMAGE_CACHE_DIR = Path(__file__).resolve().parent / ".vehicle_image_cache"
IMAGE_CACHE_DIR.mkdir(exist_ok=True)

@st.cache_data
def load_data():
    df = pd.read_csv("clean_vehicles.csv", low_memory=False)
    return df

@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    classifier = joblib.load("rf_classifier.pkl")
    reg_mpg = joblib.load("rf_regressor_comb08.pkl")
    reg_co2 = joblib.load("rf_regressor_co2.pkl")
    return scaler, classifier, reg_mpg, reg_co2

@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

@st.cache_resource
def build_rag_index(df):
    documents = []
    # Using specific constraints for sentence generation to make it fast
    # Take a sample or build sentences for all ~20k rows? 
    # Building for all can be slow on startup. We will do it for a subset or just build them.
    for idx, row in df.iterrows():
        sentences = f"The {row['year']} {row['make']} {row['model']} is a {row['VClass']} with a {row['cylinders']} cylinder {row['displ']}L engine, {row['drive']}, and a {row['trany']} transmission. It achieves a combined {row['comb08']} MPG and produces {row['co2TailpipeGpm']} g/mi of CO2, with an estimated annual fuel cost of {row['youSaveSpend']}."
        documents.append(sentences)
        
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return documents, vectorizer, tfidf_matrix


def _vehicle_image_cache_path(year, make, model_name):
    query = f"{year} {make} {model_name} car"
    query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
    return IMAGE_CACHE_DIR / f"{query_hash}.jpg"


def get_vehicle_image(year, make, model_name):
    cache_path = _vehicle_image_cache_path(year, make, model_name)
    if cache_path.exists():
        return cache_path

    query = f"{year} {make} {model_name} car"
    try:
        results = list(DDGS().images(query, max_results=5))
    except Exception:
        return None

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }

    for result in results:
        img_url = result.get("image") or result.get("thumbnail")
        if not img_url:
            continue

        try:
            response = requests.get(img_url, timeout=12, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type.lower():
                continue

            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(cache_path, format="JPEG", quality=90)
            return cache_path
        except Exception:
            continue

    return None

def main():
    st.title("Explore Fuel Economy")
    
    # 1. Session State Initialization
    if 'models_loaded' not in st.session_state:
        st.session_state['models_loaded'] = True
        st.session_state['scaler'], st.session_state['classifier'], st.session_state['reg_mpg'], st.session_state['reg_co2'] = load_models()
        st.session_state['df'] = load_data()
        st.session_state['docs'], st.session_state['vec'], st.session_state['tfidf'] = build_rag_index(st.session_state['df'])
        
    df = st.session_state['df']
    scaler = st.session_state['scaler']
    reg_mpg = st.session_state['reg_mpg']
    reg_co2 = st.session_state['reg_co2']
    classifier = st.session_state['classifier']
    
    # 2. YOLO image upload
    st.sidebar.header("Step 1: Verify car Image")
    uploaded_file = st.sidebar.file_uploader("Upload car driving image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        yolo_model = load_yolo()
        image = Image.open(uploaded_file)
        results = yolo_model(image)
        car_detected = False
        boxes = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 2: # COCO id 2 is car
                    car_detected = True
                    boxes.append(box)
                    
        if car_detected:
            st.sidebar.success("Car detected!")
            res_plotted = results[0].plot()
            st.sidebar.image(res_plotted, caption="YOLO Detection")
        else:
            st.sidebar.error("Warning: No car detected in the image.")
            
    # 3. Flexible Inputs (Year -> Make -> Model) displayed collectively
    st.sidebar.header("Step 2: Select Vehicle")
    
    if st.sidebar.button("Reset Selections"):
        st.session_state.pop("cs_year", None)
        st.session_state.pop("cs_make", None)
        st.session_state.pop("cs_model", None)
        st.rerun()

    # 1. Read current state
    cy = st.session_state.get("cs_year")
    cm = st.session_state.get("cs_make")
    cmod = st.session_state.get("cs_model")

    # 2. Build option lists restricted by active filters
    df_year = df.copy()
    if cm: df_year = df_year[df_year['make'] == cm]
    if cmod: df_year = df_year[df_year['model'] == cmod]
    valid_years = sorted(df_year['year'].dropna().unique(), reverse=True)

    df_make = df.copy()
    if cy: df_make = df_make[df_make['year'] == cy]
    if cmod: df_make = df_make[df_make['model'] == cmod]
    valid_makes = sorted(df_make['make'].dropna().unique())

    df_model = df.copy()
    if cy: df_model = df_model[df_model['year'] == cy]
    if cm: df_model = df_model[df_model['make'] == cm]
    valid_models = sorted(df_model['model'].dropna().unique())

    # 3. Eject selections if cross-filtering invalidated them
    if cy and cy not in valid_years: st.session_state.cs_year = None
    if cm and cm not in valid_makes: st.session_state.cs_make = None
    if cmod and cmod not in valid_models: st.session_state.cs_model = None

    # 4. Render selectboxes so ALL are visible at startup
    year = st.sidebar.selectbox("Year", valid_years, index=None, placeholder="Select Year...", key="cs_year")
    make = st.sidebar.selectbox("Make", valid_makes, index=None, placeholder="Select Make...", key="cs_make")
    model_name = st.sidebar.selectbox("Model", valid_models, index=None, placeholder="Select Model...", key="cs_model")

    # 5. Local RAG with Ollama (moved intentionally before the st.stop block)
    st.sidebar.write("---")
    st.sidebar.header("Step 3: Ask Ollama (RAG)")
    model_choice = st.sidebar.selectbox("Select local LLM:", ["tinyllama", "llama3.1:8b"])
    user_query = st.sidebar.text_input("Ask a question about the database or this vehicle:")
    
    if user_query:
        # Retrieve context
        vec = st.session_state['vec']
        tfidf = st.session_state['tfidf']
        docs = st.session_state['docs']
        
        query_vec = vec.transform([user_query])
        sims = cosine_similarity(query_vec, tfidf).flatten()
        top_k_indices = sims.argsort()[-5:][::-1]
        retrieved_context = [docs[i] for i in top_k_indices]
        
        with st.spinner("Ollama is thinking..."):
            try:
                prompt = f"""Use ONLY the following context to answer the user's question. 
                Context:
                {' '.join(retrieved_context)}
                
                Question: {user_query}
                """
                
                response = ollama.chat(model=model_choice, messages=[
                    {'role': 'system', 'content': 'You are a vehicle AI assistant.'},
                    {'role': 'user', 'content': prompt}
                ])
                
                st.sidebar.write("**Answer:**")
                st.sidebar.write(response['message']['content'])
                
                with st.sidebar.expander("View retrieved context"):
                    for c in retrieved_context:
                        st.write("- " + c)
            except Exception as e:
                st.sidebar.error(f"Error calling Ollama with model '{model_choice}'. Make sure Ollama is running natively and the model is pulled. {e}")

    if not year or not make or not model_name:
        st.info("👈 Please select a Year, Make, and Model via the sidebar to execute the pipeline.")
        st.stop()
    
    # Trim check
    selected_subset = df[(df['year'] == year) & (df['make'] == make) & (df['model'] == model_name)]
    if len(selected_subset) > 1:
        # Create trim keys
        selected_subset['trim_label'] = selected_subset['drive'].fillna('Unknown Drive') + " — " + selected_subset['trany'].fillna('Unknown Transmission')
        trim_options = selected_subset['trim_label'].unique()
        trim = st.sidebar.selectbox("Trim (Drive — Trany)", trim_options)
        selected_vehicle = selected_subset[selected_subset['trim_label'] == trim].iloc[0]
    else:
        st.sidebar.info("Only one trim available for this model.")
        if len(selected_subset) == 1:
            selected_vehicle = selected_subset.iloc[0]
        else:
            st.warning("No data found for selection.")
            st.stop()
            
    st.subheader(f"Selected: {year} {make} {model_name}")
    
    with st.spinner("Finding car image..."):
        image_path = get_vehicle_image(year, make, model_name)
        if image_path is not None and image_path.exists():
            st.image(str(image_path), caption=f"{year} {make} {model_name}", width=500)
        else:
            st.info("No vehicle image could be retrieved right now.")
            
    # Show measured targets
    col1, col2 = st.columns(2)
    measured_mpg = selected_vehicle['comb08']
    measured_co2 = selected_vehicle['co2TailpipeGpm']
    
    col1.metric("Measured Combined MPG", f"{measured_mpg:.1f}")
    col2.metric("Measured Tailpipe CO2 (g/mi)", f"{measured_co2:.1f}")
    
    # 4. Predictions
    # To predict, we must reconstruct the feature vector!
    st.write("---")
    st.subheader("Model Predictions")
    
    # Note: re-creating the exact feature matrix for a single row can be complex if we don't have the exact columns.
    # We will use the fact that we can recreate df_encoded.
    # Fortunately, the prompt allows us to just run inference or use the prebuilt features. Let's build a small helper string 
    # indicating we will mock the exact pipeline.
    # Given the complexity constraints, if live-encoding fails (missing columns from train set), we will just load expected columns. Let's do it safely:
    # Actually, the user asked to train the model, save it, and predict. In the apmonitor dashboard, doing this requires the original feature_names_in_.
    try:
        features = scaler.feature_names_in_
        # Reconstruct the row dataframe
        row_df = pd.DataFrame([selected_vehicle])
        
        # Categorical groupings
        row_df['trany_group'] = 'Other'
        if 'Manual' in str(row_df['trany'].values[0]): row_df['trany_group'] = 'Manual'
        elif 'CVT' in str(row_df['trany'].values[0]): row_df['trany_group'] = 'Automatic CVT'
        elif 'Automatic' in str(row_df['trany'].values[0]): row_df['trany_group'] = 'Automatic'
        
        row_df['VClass_group'] = 'Other'
        v = str(row_df['VClass'].values[0]).lower()
        if 'car' in v or 'compact' in v or 'sedan' in v: row_df['VClass_group'] = 'Car'
        elif 'suv' in v or 'utility' in v: row_df['VClass_group'] = 'SUV'
        elif 'truck' in v or 'pickup' in v: row_df['VClass_group'] = 'Truck'
        elif 'van' in v: row_df['VClass_group'] = 'Van'
        
        row_df['has_turbo'] = 1 if 'turbo' in str(row_df['model'].values[0]).lower() else 0
        row_df['has_supercharger'] = 1 if 'supercharger' in str(row_df['model'].values[0]).lower() else 0
        row_df['is_startstop'] = 1 if 'y' in str(row_df['startStop'].values[0]).lower() else 0
        row_df['is_awd'] = 1 if '4-wheel' in str(row_df['drive'].values[0]).lower() or 'all-wheel' in str(row_df['drive'].values[0]).lower() or '4wd' in str(row_df['drive'].values[0]).lower() or 'awd' in str(row_df['drive'].values[0]).lower() else 0
        row_df['gear_count'] = int(str(row_df['trany'].values[0]).split()[-1].replace('spd','').replace('(','').replace(')','')) if 'spd' in str(row_df['trany'].values[0]) else 0

        # Create one-hot variables
        row_encoded = pd.DataFrame(columns=features)
        row_encoded.loc[0] = 0 # fill with 0s
        
        # For numeric cols, copy them
        for col in features:
            if col in row_df.columns and pd.api.types.is_numeric_dtype(row_df[col]):
                row_encoded[col] = row_df[col].fillna(0).values[0]
                
        # Fill specific dummies
        if f"VClass_group_{row_df['VClass_group'].values[0]}" in features:
            row_encoded[f"VClass_group_{row_df['VClass_group'].values[0]}"] = 1
        if f"drive_{row_df['drive'].values[0]}" in features:
            row_encoded[f"drive_{row_df['drive'].values[0]}"] = 1
        if f"fuelType1_{row_df['fuelType1'].values[0]}" in features:
            row_encoded[f"fuelType1_{row_df['fuelType1'].values[0]}"] = 1
        if f"trany_group_{row_df['trany_group'].values[0]}" in features:
            row_encoded[f"trany_group_{row_df['trany_group'].values[0]}"] = 1
            
        row_encoded['has_turbo'] = row_df['has_turbo'].values[0]
        row_encoded['has_supercharger'] = row_df['has_supercharger'].values[0]
        row_encoded['is_startstop'] = row_df['is_startstop'].values[0]
        row_encoded['is_awd'] = row_df['is_awd'].values[0]
        row_encoded['gear_count'] = row_df['gear_count'].values[0]
        
        X_test = scaler.transform(row_encoded)
        
        pred_mpg = reg_mpg.predict(X_test)[0]
        pred_co2 = reg_co2.predict(X_test)[0]
        pred_eff = classifier.predict(X_test)[0]
        
        col3, col4, col5 = st.columns(3)
        col3.metric("Predicted MPG", f"{pred_mpg:.1f}", f"{(pred_mpg - measured_mpg):.1f} error")
        col4.metric("Predicted CO2", f"{pred_co2:.1f}", f"{(pred_co2 - measured_co2):.1f} error")
        col5.metric("High Efficiency Classification", "Yes" if pred_eff == 1 else "No")
        
        # Show a quick bar chart comparing measured vs predicted MPG
        chart_data = pd.DataFrame({
            "Source": ["Measured", "Predicted"],
            "MPG": [measured_mpg, pred_mpg]
        })
        st.bar_chart(chart_data.set_index("Source"))
        
    except Exception as e:
        st.warning(f"Prediction failed for this unstandardized trim: {e}")

if __name__ == "__main__":
    main()
