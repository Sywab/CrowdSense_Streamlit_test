# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from datetime import datetime

# Import your prediction functions
from recto_predict import predict_recto_week
from legarda_predict import predict_legarda_week
from pureza_predict import predict_pureza_week
from vmapa_predict import predict_vmapa_week
from jruiz_predict import predict_jruiz_week
from gilmore_predict import predict_gilmore_week
from bettygobelmonte_predict import predict_bettygobelmonte_week
from araneta_cubao_predict import predict_araneta_cubao_week
from anonas_predict import predict_anonas_week
from katipunan_predict import predict_katipunan_week
from santolan_predict import predict_santolan_week
from marikina_pasig_predict import predict_marikina_pasig_week
from antipolo_predict import predict_antipolo_week

# -------------------------------
# Helper functions
# -------------------------------

hours_per_block = [2, 2, 2, 2, 8, 8, 2, 2, 3, 3]

def process_station(predictions, time_slots, today_short, station_name):
    today = predictions.get(today_short, None)
    avg_per_hour = None
    peak_block = None
    peak_density = None
    avg_density = None
    peak_density_per_hour = None
    peak_block_per_hour = None

    if today is not None and len(today) > 0:
        avg_per_hour = [round(val / hrs, 2) for val, hrs in zip(today, hours_per_block)]
        arr = np.array(today)
        peak_idx = int(arr.argmax())
        peak_block = time_slots[peak_idx]
        peak_density = float(arr.max())
        avg_density = float(arr.mean())
        arr_per_hour = np.array(avg_per_hour)
        peak_idx_per_hour = int(arr_per_hour.argmax())
        peak_block_per_hour = time_slots[peak_idx_per_hour]
        peak_density_per_hour = float(arr_per_hour.max())
    return {
        f"{station_name}_today": today,
        f"{station_name}_avg_per_hour": avg_per_hour,
        f"{station_name}_time_slots": time_slots,
        f"{station_name}_today_short": today_short,
        f"{station_name}_peak_block": peak_block,
        f"{station_name}_peak_density": peak_density,
        f"{station_name}_avg_density": avg_density,
        f"{station_name}_peak_density_per_hour": peak_density_per_hour,
        f"{station_name}_peak_block_per_hour": peak_block_per_hour
    }

def _pretty_block(label):
    if not label:
        return None
    parts = label.split('-')
    if len(parts) == 2:
        a, b = parts
        a = a.upper()
        b = b.upper()
        def fmt(t):
            if t.endswith('AM') or t.endswith('PM'):
                return t[:-2] + ' ' + t[-2:]
            return t
        return f"{fmt(a)} - {fmt(b)}"
    return label

# -------------------------------
# Sidebar navigation
# -------------------------------

st.sidebar.title("CrowdSense Dashboard")
page = st.sidebar.selectbox("Go to", ["Dashboard", "Analytics", "Upload CSV", "Admin Login"])

# -------------------------------
# Admin login state
# -------------------------------
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"  # change this in production

# -------------------------------
# Dashboard page
# -------------------------------
if page == "Dashboard":
    st.title("LRT-2 Stations - Daily Predictions")

    station_funcs = [
        ("recto", predict_recto_week),
        ("legarda", predict_legarda_week),
        ("pureza", predict_pureza_week),
        ("vmapa", predict_vmapa_week),
        ("jruiz", predict_jruiz_week),
        ("gilmore", predict_gilmore_week),
        ("bettygobelmonte", predict_bettygobelmonte_week),
        ("araneta_cubao", predict_araneta_cubao_week),
        ("anonas", predict_anonas_week),
        ("katipunan", predict_katipunan_week),
        ("santolan", predict_santolan_week),
        ("marikina_pasig", predict_marikina_pasig_week),
        ("antipolo", predict_antipolo_week)
    ]

    context = {}
    for name, func in station_funcs:
        try:
            preds, time_slots, today_short = func()
            context.update(process_station(preds, time_slots, today_short, name))
        except Exception as e:
            st.warning(f"Error loading predictions for {name}: {e}")

    # Display per-station peak density
    st.subheader("Peak Densities per Hour")
    peak_data = {name: context.get(f"{name}_peak_density_per_hour", 0) for name, _ in station_funcs}
    st.bar_chart(pd.DataFrame(list(peak_data.items()), columns=["Station", "Peak Density"]).set_index("Station"))

    # Display most common peak block
    peak_blocks = [context.get(f"{name}_peak_block_per_hour") for name, _ in station_funcs if context.get(f"{name}_peak_block_per_hour")]
    if peak_blocks:
        most_common_block, count = Counter(peak_blocks).most_common(1)[0]
        st.write(f"Most common peak block: **{_pretty_block(most_common_block)}** ({count} stations)")

# -------------------------------
# Analytics page
# -------------------------------
elif page == "Analytics":
    st.title("Analytics & Ridership Trends")
    st.write("Here you can display year-over-year comparisons, monthly trends, etc.")
    st.info("You will need to integrate your plotting functions from `year_over_year_comparison.py` and `monthly_density_trend.py` here.")

# -------------------------------
# Upload CSV page
# -------------------------------
elif page == "Upload CSV":
    st.title("Upload CSV Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        save_path = os.path.join("static", uploaded_file.name)
        df.to_csv(save_path, index=False)
        st.success(f"File saved to {save_path}")

# -------------------------------
# Admin login page
# -------------------------------
elif page == "Admin Login":
    if st.session_state.admin_logged_in:
        st.success("Already logged in as admin.")
        st.write("Access your uploaded CSVs below:")
        if os.path.exists("static"):
            csv_files = [f for f in os.listdir("static") if f.endswith(".csv")]
            st.write(csv_files)
    else:
        st.subheader("Admin Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.success("Login successful!")
            else:
                st.error("Invalid username or password")
