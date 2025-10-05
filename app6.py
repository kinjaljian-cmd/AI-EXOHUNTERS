import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --------------------------
# Helper Functions
# --------------------------
def safe_flatten_cell(x):
    if isinstance(x, (list, tuple, np.ndarray, dict)):
        return str(x)
    return x

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.applymap(safe_flatten_cell)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
    return df

def enforce_arrow_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df

def standardize_columns(df: pd.DataFrame, mission: str) -> pd.DataFrame:
    col_map = {
        "koi": {
            "kepoi_name":"Name",
            "koi_disposition":"Disposition",
            "koi_prad":"Radius",
            "koi_period":"OrbitalPeriod",
            "koi_srad":"StellarRadius",
            "koi_steff":"StellarTeff",
            "ra":"RA",
            "dec":"DEC",
            "koi_insol":"Insolation"
        },
        "toi": {
            "toi":"Name",
            "tfopwg_disp":"Disposition",
            "pl_rade":"Radius",
            "pl_orbper":"OrbitalPeriod",
            "st_rad":"StellarRadius",
            "st_teff":"StellarTeff",
            "ra":"RA",
            "dec":"DEC",
            "pl_insol":"Insolation"
        },
        "k2": {
            "name":"Name",
            "disposition":"Disposition",
            "radius":"Radius",
            "period":"OrbitalPeriod",
            "st_rad":"StellarRadius",
            "st_teff":"StellarTeff",
            "ra":"RA",
            "dec":"DEC",
            "insolation":"Insolation"
        }
    }

    df = df.rename(columns=col_map.get(mission, {}))
    df["Mission"] = mission.upper()

    # Required columns
    required_cols = ["Name","Disposition","Radius","OrbitalPeriod",
                     "StellarRadius","StellarTeff","RA","DEC","Insolation","Mission"]
    for col in required_cols:
        if col not in df.columns:
            if col in ["Radius","OrbitalPeriod","StellarRadius","StellarTeff","RA","DEC","Insolation"]:
                df[col] = np.nan
            else:
                df[col] = "Unknown"

    # Force numeric conversion and fill missing values with small numbers
    for col in ["Radius","OrbitalPeriod","StellarRadius","StellarTeff","RA","DEC","Insolation"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Fill NaN with median if median exists, else small default value
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.1)

    return df

def classify_habitability(row):
    try:
        radius = float(row["Radius"])
        insolation = float(row["Insolation"])
        if radius < 2.0 and 0.3 < insolation < 1.5:
            return "Earth-like"
        elif radius < 10:
            return "Super-Earth / Mini-Neptune"
        else:
            return "Gas Giant"
    except:
        return "Unknown"

# --------------------------
# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="AI EXOHUNTERS", layout="wide")
st.title("🚀 AI EXOHUNTERS")

st.markdown("""
Explore exoplanets with interactive visuals, habitability insights, and ML simulations.  

**Tips:**  
- Hover over planets for detailed info  
- Habitability is color-coded  
- Simulation tab predicts planet disposition using ML
""")

# File Upload or Default Dataset
# --------------------------
uploaded_files = st.file_uploader("Upload KOI, TOI, and K2 CSV files", accept_multiple_files=True, type=["csv"])
dfs = []

if uploaded_files:
    for file in uploaded_files:
        fname = file.name.lower()
        if "koi" in fname:
            df = pd.read_csv(file); df = clean_dataframe(df); dfs.append(standardize_columns(df, "koi"))
        elif "toi" in fname:
            df = pd.read_csv(file); df = clean_dataframe(df); dfs.append(standardize_columns(df, "toi"))
        elif "k2" in fname:
            df = pd.read_csv(file); df = clean_dataframe(df); dfs.append(standardize_columns(df, "k2"))

if dfs:
    df_all = pd.concat(dfs, ignore_index=True)
else:
    st.info("No files uploaded. Using default demo dataset.")
    np.random.seed(42)
    df_all = pd.DataFrame({
        "Name": [f"Planet {i}" for i in range(1,21)],
        "Disposition": np.random.choice(["Confirmed","Candidate","False Positive"], 20),
        "Radius": np.random.uniform(0.5, 15, 20),
        "OrbitalPeriod": np.random.uniform(0.5, 500, 20),
        "StellarRadius": np.random.uniform(0.5, 2.0, 20),
        "StellarTeff": np.random.uniform(3000,7000,20),
        "RA": np.random.uniform(0,360,20),
        "DEC": np.random.uniform(-90,90,20),
        "Insolation": np.random.uniform(0.1, 10, 20),
        "Mission": np.random.choice(["KOI","TOI","K2"], 20)
    })

df_all["Habitability"] = df_all.apply(classify_habitability, axis=1)

# --------------------------
# Tabs
# --------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Filters", "Visualizations", "Sky Map", "Simulation"])

# --------------------------
# Tab 1: Overview
# --------------------------
with tab1:
    st.header("📊 Dataset Overview")
    st.dataframe(enforce_arrow_compatibility(df_all.head(50)))
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Planets", len(df_all))
    col2.metric("Missions", len(df_all["Mission"].unique()))
    col3.metric("Dispositions", len(df_all["Disposition"].unique()) if "Disposition" in df_all.columns else 0)
    col4.metric("Avg Radius (Earth)", round(df_all["Radius"].mean(),2))

# --------------------------
# --------------------------
# Tab 2: Filters
# --------------------------
with tab2:
    st.header("🔍 Filter Planets")
    mission = st.multiselect("Select Mission", options=df_all["Mission"].unique(), default=list(df_all["Mission"].unique()))
    disp = st.multiselect("Select Disposition", options=df_all["Disposition"].unique(), default=list(df_all["Disposition"].unique()))
    
    # Ensure numeric sliders are float and safe
    min_rad_val = float(df_all["Radius"].min())
    max_rad_val = float(df_all["Radius"].max())
    if min_rad_val == max_rad_val:
        max_rad_val += 0.01  # small increment to allow slider

    min_per_val = float(df_all["OrbitalPeriod"].min())
    max_per_val = float(df_all["OrbitalPeriod"].max())
    if min_per_val == max_per_val:
        max_per_val += 0.01

    # Sliders
    min_rad, max_rad = st.slider(
        "Planet Radius (Earth radii)", min_rad_val, max_rad_val, (min_rad_val, max_rad_val)
    )
    min_per, max_per = st.slider(
        "Orbital Period (days)", min_per_val, max_per_val, (min_per_val, max_per_val)
    )

    # Filter DataFrame
    df_filtered = df_all[
        (df_all["Mission"].isin(mission)) &
        (df_all["Disposition"].isin(disp)) &
        (df_all["Radius"].between(min_rad, max_rad)) &
        (df_all["OrbitalPeriod"].between(min_per, max_per))
    ]

    st.write(f"Filtered Planets: {len(df_filtered)}")
    st.dataframe(enforce_arrow_compatibility(df_filtered.head(50)))

# --------------------------
# Tab 3: Visualizations (Enhanced for Layman)
# --------------------------
with tab3:
    st.header("📊 Exoplanet Data Visualizations")

    if df_filtered.empty:
        st.warning("No data to visualize with current filters.")
    else:
        # 1️⃣ Scatter Plot: OrbitalPeriod vs Radius
        st.subheader("🌌 Orbital Period vs Planet Radius")
        st.markdown("💡 **What this shows:** How big planets are vs. how long they take to orbit their star. "
                    "Hover over planets to see their details like size, star type, and mission.")
        fig1 = px.scatter(
            df_filtered, x="OrbitalPeriod", y="Radius", color="Habitability",
            hover_data=["Name","Disposition","Mission","StellarRadius","StellarTeff"],
            template="plotly_dark", size_max=20, symbol="Mission"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 2️⃣ Habitability Distribution
        st.subheader("📊 Habitability Distribution")
        st.markdown("💡 **Explanation:** Counts of planets by habitability category. "
                    "Colors indicate potential habitability: Earth-like (green) is most likely habitable.")
        fig2 = px.histogram(df_filtered, x="Habitability", color="Habitability",
                            template="plotly_dark", text_auto=True,
                            color_discrete_map={"Earth-like":"green",
                                                "Super-Earth / Mini-Neptune":"orange",
                                                "Gas Giant":"blue",
                                                "Unknown":"gray"})
        st.plotly_chart(fig2, use_container_width=True)

        # 3️⃣ Box Plot: Radius by Habitability
        st.subheader("📦 Planet Radius Distribution by Habitability")
        st.markdown("💡 **Explanation:** Shows the range of planet sizes in each habitability category. "
                    "Dots are individual planets; box shows average and spread.")
        fig3 = px.box(df_filtered, x="Habitability", y="Radius", color="Habitability",
                      template="plotly_dark", points="all", hover_data=["Name","Mission"])
        st.plotly_chart(fig3, use_container_width=True)

        # 4️⃣ Violin Plot: Insolation by Habitability
        st.subheader("🎻 Insolation Received by Planets")
        st.markdown("💡 **Explanation:** How much energy planets receive from their star. "
                    "Ideal range (Earth-like) is 0.3-1.5 times Earth's sunlight.")
        fig4 = px.violin(df_filtered, x="Habitability", y="Insolation", color="Habitability",
                         box=True, points="all", template="plotly_dark", hover_data=["Name","Mission"])
        st.plotly_chart(fig4, use_container_width=True)

        # 5️⃣ Sunburst Chart
        st.subheader("🌞 Sunburst: Habitability, Disposition & Mission")
        st.markdown("💡 **Explanation:** Nested view: shows proportion of planets by habitability, then by disposition, then by mission.")
        fig5 = px.sunburst(df_filtered, path=["Habitability","Disposition","Mission"], values="Radius",
                           color="Radius", color_continuous_scale="Viridis", template="plotly_dark")
        st.plotly_chart(fig5, use_container_width=True)

        # 6️⃣ 3D Scatter Plot
        st.subheader("🪐 3D Scatter: Orbital Period, Planet Radius & Stellar Radius")
        st.markdown("💡 **Explanation:** 3D view of planet size vs orbit vs star size. Hover to see planet name, mission, and habitability.")
        fig6 = px.scatter_3d(df_filtered, x="OrbitalPeriod", y="Radius", z="StellarRadius",
                             color="Habitability", size="Radius",
                             hover_data=["Name","Disposition","Mission"], template="plotly_dark")
        st.plotly_chart(fig6, use_container_width=True)

        # 7️⃣ Density Heatmap
        st.subheader("🔥 Density Heatmap: Planet Radius vs Orbital Period")
        st.markdown("💡 **Explanation:** Shows where most planets cluster in size and orbital period. Darker = more planets.")
        fig7 = px.density_heatmap(df_filtered, x="OrbitalPeriod", y="Radius", nbinsx=30, nbinsy=30,
                                  template="plotly_dark", color_continuous_scale="Viridis")
        st.plotly_chart(fig7, use_container_width=True)

# Tab 4: Sky Map
# --------------------------
with tab4:
    st.header("🌌 Sky Map (RA / DEC)")
    if df_filtered.empty:
        st.warning("Sky map cannot be drawn. No coordinates available.")
    else:
        df_map = df_filtered.copy()
        df_map["PointSize"] = 5 + (df_map["Radius"] / df_map["Radius"].max()) * 20
        fig_sky = px.scatter(df_map, x="RA", y="DEC", color="Habitability", size="PointSize",
                             hover_data=["Name","Radius","OrbitalPeriod","Disposition","Mission"],
                             title="Exoplanet Sky Map", size_max=35, template="plotly_dark")
        fig_sky.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="RA", yaxis_title="DEC")
        st.plotly_chart(fig_sky, use_container_width=True)

# --------------------------
# Tab 5: Simulation
# --------------------------
with tab5:
    st.header("🤖 Planet Discovery Simulation")
    X = df_all[["Radius","OrbitalPeriod","StellarRadius","StellarTeff","Insolation"]].fillna(0)
    y = df_all["Disposition"].fillna("Unknown")

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        st.subheader("Model Performance")
        st.text(classification_report(y_test, preds))

        n_sim = st.slider("Number of simulated planets", 1, 20, 5)
        sim_data = pd.DataFrame({
            "Radius": np.random.uniform(0.5,15,n_sim),
            "OrbitalPeriod": np.random.uniform(0.5,500,n_sim),
            "StellarRadius": np.random.uniform(0.5,2.0,n_sim),
            "StellarTeff": np.random.uniform(3000,7000,n_sim),
            "Insolation": np.random.uniform(0.1,10,n_sim)
        })
        sim_data["PredictedDisposition"] = clf.predict(sim_data)
        sim_data["Habitability"] = sim_data.apply(classify_habitability, axis=1)
        st.dataframe(enforce_arrow_compatibility(sim_data))

    except Exception as e:
        st.error(f"ML Simulation failed: {e}")
