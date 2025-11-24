import streamlit as st
from werkzeug.utils import secure_filename
import os
import pandas as pd
from year_over_year_comparison import save_historical_comparison_plot
from monthly_density_trend import monthly_density_trend_plot
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
from datetime import datetime
from collections import Counter
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Change this!

# Admin credentials (in production, use environment variables or database)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'  # Change this!

# Upload configuration
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    hours_per_block = [2, 2, 2, 2, 8, 8, 2, 2, 3, 3]

    # Helper function for all stations
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

    # Get predictions for all stations
    context = {}
    context.update(process_station(*predict_recto_week(), "recto"))
    context.update(process_station(*predict_legarda_week(), "legarda"))
    context.update(process_station(*predict_pureza_week(), "pureza"))
    context.update(process_station(*predict_vmapa_week(), "vmapa"))
    context.update(process_station(*predict_jruiz_week(), "jruiz"))
    context.update(process_station(*predict_gilmore_week(), "gilmore"))
    context.update(process_station(*predict_bettygobelmonte_week(), "bettygobelmonte"))
    context.update(process_station(*predict_araneta_cubao_week(), "araneta_cubao"))
    context.update(process_station(*predict_anonas_week(), "anonas"))
    context.update(process_station(*predict_katipunan_week(), "katipunan"))
    context.update(process_station(*predict_santolan_week(), "santolan"))
    context.update(process_station(*predict_marikina_pasig_week(), "marikina_pasig"))
    context.update(process_station(*predict_antipolo_week(), "antipolo"))

    # Add shared context
    context.update({
        "now": datetime.now(),
        "zip": zip,
        "max": max
    })

    # ...existing code...
    station_prefixes = [
        "recto", "legarda", "pureza", "vmapa", "jruiz", "gilmore",
        "bettygobelmonte", "araneta_cubao", "anonas", "katipunan",
        "santolan", "marikina_pasig", "antipolo"
    ]

    # Count stations at "High" (75-89.999...) and "Critical" (>=90) density
    high_count = 0
    very_high_count = 0
    critical_count = 0

    for prefix in station_prefixes:
        val = context.get(f"{prefix}_peak_density_per_hour")
        if val is None:
            continue
        try:
            v = float(val)
        except Exception:
            continue
        if v >= 750:
            critical_count += 1
        elif v >= 1000:
            very_high_count += 1
        elif v >= 1250:
            high_count += 1

    high_very_critical_total = high_count + very_high_count + critical_count

    context["high_count"] = high_count
    context["very_high_count"] = very_high_count
    context["critical_count"] = critical_count
    context["high_very_critical_total"] = high_very_critical_total

    context["high_density_count"] = high_very_critical_total

    vals = []
    for prefix in station_prefixes:
        v = context.get(f"{prefix}_peak_density_per_hour")
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    context["average_density_all"] = (sum(vals) / len(vals)) if vals else None

    peak_blocks = []
    for prefix in station_prefixes:
        blk = context.get(f"{prefix}_peak_block_per_hour")
        if blk:
            block_label = str(blk).split()[0]
            peak_blocks.append(block_label)

    most_common_peak_block = None
    most_common_count = 0
    if peak_blocks:
        most_common_peak_block, most_common_count = Counter(peak_blocks).most_common(1)[0]

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

    context["most_common_peak_block"] = most_common_peak_block
    context["most_common_peak_block_pretty"] = _pretty_block(most_common_peak_block)
    context["most_common_peak_block_count"] = most_common_count

    return render_template('dashboard.html', **context)

@app.route('/mapsroutes')
def mapsroutes():
    return render_template('mapsroutes.html')

@app.route('/analytics')
def analytics():
    save_historical_comparison_plot()
    monthly_density_trend_plot()

    hours_per_block = [2, 2, 2, 2, 8, 8, 2, 2, 3, 3]

    # list of station name and corresponding predict function (ordered from Recto to Antipolo)
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

    peak_blocks = []
    busiest_station_name = None
    busiest_station_value = None

    for name, func in station_funcs:
        try:
            preds, time_slots, today_short = func()
        except Exception:
            continue
        if not preds:
            continue
        today_pred = preds.get(today_short)
        if today_pred is None:
            continue
        try:
            # compute avg per hour and find peak block index & value
            avg_per_hour = [val / hrs for val, hrs in zip(today_pred, hours_per_block)]
            arr_per_hour = np.array(avg_per_hour)
            peak_idx = int(arr_per_hour.argmax())
            peak_value = float(arr_per_hour[peak_idx])
            block_label = time_slots[peak_idx]
            peak_blocks.append(str(block_label).split()[0])
            # update busiest station
            if busiest_station_value is None or peak_value > busiest_station_value:
                busiest_station_value = peak_value
                # prettify station name for display
                busiest_station_name = name.replace('_', ' ').title()
        except Exception:
            continue

    most_common_peak_block = None
    most_common_count = 0
    if peak_blocks:
        most_common_peak_block, most_common_count = Counter(peak_blocks).most_common(1)[0]

    def _pretty_block(label):
        if not label:
            return None
        parts = label.split('-')
        if len(parts) == 2:
            a, b = parts
            return f"{a.upper()} - {b.upper()}"
        return label

    most_common_peak_block_pretty = _pretty_block(most_common_peak_block)

    # --- compute yearly totals (2023 vs 2024) across all station CSVs ---
    time_slots = [
        '5am-7am entry', '5am-7am exit',
        '7am-9am entry', '7am-9am exit',
        '9am-5pm entry', '9am-5pm exit',
        '5pm-7pm entry', '5pm-7pm exit',
        '7pm-10pm entry', '7pm-10pm exit'
    ]

    total_2023 = 0.0
    total_2024 = 0.0

    for station_name, _ in station_funcs:
        csv_path = os.path.join('static', f'{station_name}.csv')
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, parse_dates=['date'])
        except Exception:
            continue
        # ensure required columns exist
        cols = [c for c in time_slots if c in df.columns]
        if not cols:
            continue
        # sum entries+exits for each row, then sum rows filtered by year
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            total_2023 += df.loc[df['year'] == 2023, cols].sum().sum()
            total_2024 += df.loc[df['year'] == 2024, cols].sum().sum()

    # --- compute yearly totals from LRT_2_Annual_Ridership_2022-2024.csv ---
    ridership_csv = os.path.join('static', 'LRT_2_Annual_Ridership_2022-2024.csv')
    ridership_total_2022 = ridership_total_2023 = ridership_total_2024 = 0.0

    if os.path.exists(ridership_csv):
        try:
            rdf = pd.read_csv(ridership_csv)
            cols_lower = [c.lower() for c in rdf.columns]
            if 'year' in cols_lower and ('ridership' in cols_lower or 'total' in cols_lower):
                ycol = rdf.columns[cols_lower.index('year')]
                if 'ridership' in cols_lower:
                    vcol = rdf.columns[cols_lower.index('ridership')]
                else:
                    vcol = rdf.columns[cols_lower.index('total')]
                grp = rdf.groupby(rdf[ycol].astype(int))[vcol].sum()
                ridership_total_2022 = float(grp.get(2022, 0))
                ridership_total_2023 = float(grp.get(2023, 0))
                ridership_total_2024 = float(grp.get(2024, 0))
            elif '2022' in rdf.columns or '2023' in rdf.columns or '2024' in rdf.columns:
                if '2022' in rdf.columns:
                    ridership_total_2022 = float(rdf['2022'].dropna().sum())
                if '2023' in rdf.columns:
                    ridership_total_2023 = float(rdf['2023'].dropna().sum())
                if '2024' in rdf.columns:
                    ridership_total_2024 = float(rdf['2024'].dropna().sum())
            else:
                # fallback: parse date/year and sum numeric cols by year
                if 'date' in cols_lower:
                    dcol = rdf.columns[cols_lower.index('date')]
                    rdf[dcol] = pd.to_datetime(rdf[dcol], errors='coerce')
                    rdf['__year'] = rdf[dcol].dt.year
                elif 'month' in cols_lower and 'year' in cols_lower:
                    ycol = rdf.columns[cols_lower.index('year')]
                    rdf['__year'] = rdf[ycol].astype(int)
                numeric = rdf.select_dtypes(include='number').columns.tolist()
                if numeric and '__year' in rdf.columns:
                    ridership_total_2022 = float(rdf.loc[rdf['__year'] == 2022, numeric].sum().sum())
                    ridership_total_2023 = float(rdf.loc[rdf['__year'] == 2023, numeric].sum().sum())
                    ridership_total_2024 = float(rdf.loc[rdf['__year'] == 2024, numeric].sum().sum())
        except Exception:
            pass

    # use ridership CSV totals if available, otherwise fallback to station CSV totals computed earlier
    final_2023 = ridership_total_2023 if ridership_total_2023 else total_2023
    final_2024 = ridership_total_2024 if ridership_total_2024 else total_2024

    yearly_pct_change = None
    if final_2023 and final_2023 != 0:
        yearly_pct_change = round(((final_2024 - final_2023) / final_2023) * 100, 2)

    # --- generate station distribution plot ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    station_names = []
    station_totals = []
    
    for station_name, func in station_funcs:
        try:
            preds, time_slots, today_short = func()
            today_pred = preds.get(today_short)
            if today_pred is not None:
                # sum all entries + exits for the day
                daily_total = sum(today_pred)
                station_names.append(station_name.replace('_', ' ').title())
                station_totals.append(daily_total)
            else:
                station_names.append(station_name.replace('_', ' ').title())
                station_totals.append(0)
        except Exception:
            station_names.append(station_name.replace('_', ' ').title())
            station_totals.append(0)
    
    # create bar chart
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(station_names)), station_totals, color='#30BCB1')
    plt.xlabel('LRT-2 Stations (Recto to Antipolo)')
    plt.ylabel('Total Daily Ridership')
    plt.title('Station Distribution - Total Ridership per Station (Today)')
    plt.xticks(range(len(station_names)), station_names, rotation=45, ha='right')
    
    # add value labels on bars
    for i, (bar, total) in enumerate(zip(bars, station_totals)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(station_totals)*0.01,
                f'{int(total):,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('static/images/station_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- generate daily peak hours pattern plot for analytics ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    station_names = []
    peak_densities = []
    
    for station_name, func in station_funcs:
        try:
            preds, time_slots, today_short = func()
            today_pred = preds.get(today_short)
            if today_pred is not None:
                # compute avg per hour and find peak density
                avg_per_hour = [val / hrs for val, hrs in zip(today_pred, hours_per_block)]
                peak_density = max(avg_per_hour)
                station_names.append(station_name.replace('_', ' ').title())
                peak_densities.append(peak_density)
            else:
                station_names.append(station_name.replace('_', ' ').title())
                peak_densities.append(0)
        except Exception:
            station_names.append(station_name.replace('_', ' ').title())
            peak_densities.append(0)
    
    # create bar chart for peak densities
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(station_names)), peak_densities, color='#153252')
    plt.xlabel('LRT-2 Stations (Recto to Antipolo)')
    plt.ylabel('Peak Density (per hour)')
    plt.title('Daily Peak Hours Pattern - Peak Density per Station (Today)')
    plt.xticks(range(len(station_names)), station_names, rotation=45, ha='right')
    
    # add value labels on bars
    for i, (bar, density) in enumerate(zip(bars, peak_densities)):
        if density > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(peak_densities)*0.01,
                    f'{density:.1f}', ha='center', va='bottom', fontsize=9)
    
    # add horizontal lines for density thresholds
    max_density = max(peak_densities) if peak_densities else 100
    plt.axhline(y=max_density*0.3, color='green', linestyle='--', alpha=0.7, label='Low threshold')
    plt.axhline(y=max_density*0.55, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
    plt.axhline(y=max_density*0.75, color='red', linestyle='--', alpha=0.7, label='High threshold')
    plt.axhline(y=max_density*0.9, color='darkred', linestyle='--', alpha=0.7, label='Critical threshold')
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('static/images/daily_peak_pattern.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- compute monthly change from LRT_2_Annual_Ridership_2022-2024.csv ---
    monthly_pct_change = None
    ridership_csv = os.path.join('static', 'LRT_2_Annual_Ridership_2022-2024.csv')
    
    if os.path.exists(ridership_csv):
        try:
            rdf = pd.read_csv(ridership_csv)
            
            # Case A: columns named '2024' with monthly values (12 rows)
            if '2024' in rdf.columns:
                monthly_2024 = rdf['2024'].dropna().tolist()
                if len(monthly_2024) >= 2:
                    # compute month-to-month changes and average them
                    monthly_changes = []
                    for i in range(1, len(monthly_2024)):
                        prev_month = monthly_2024[i-1]
                        curr_month = monthly_2024[i]
                        if prev_month and prev_month != 0:
                            change = ((curr_month - prev_month) / prev_month) * 100
                            monthly_changes.append(change)
                    if monthly_changes:
                        monthly_pct_change = round(sum(monthly_changes) / len(monthly_changes), 1)
            
            # Case B: date column with 2024 data
            elif 'date' in rdf.columns.str.lower():
                date_col = [c for c in rdf.columns if 'date' in c.lower()][0]
                rdf[date_col] = pd.to_datetime(rdf[date_col], errors='coerce')
                rdf['year'] = rdf[date_col].dt.year
                rdf['month'] = rdf[date_col].dt.month
                
                # get 2024 data grouped by month
                data_2024 = rdf[rdf['year'] == 2024]
                if not data_2024.empty:
                    numeric_cols = data_2024.select_dtypes(include='number').columns.tolist()
                    if numeric_cols:
                        monthly_totals = data_2024.groupby('month')[numeric_cols].sum().sum(axis=1)
                        if len(monthly_totals) >= 2:
                            monthly_changes = []
                            for i in range(1, len(monthly_totals)):
                                prev = monthly_totals.iloc[i-1]
                                curr = monthly_totals.iloc[i]
                                if prev and prev != 0:
                                    change = ((curr - prev) / prev) * 100
                                    monthly_changes.append(change)
                            if monthly_changes:
                                monthly_pct_change = round(sum(monthly_changes) / len(monthly_changes), 1)
        except Exception:
            pass

    # pass both sources if you want to inspect in template
    return render_template(
        'analytics.html',
        most_common_peak_block=most_common_peak_block,
        most_common_peak_block_pretty=most_common_peak_block_pretty,
        most_common_peak_block_count=most_common_count,
        busiest_station_name=busiest_station_name,
        busiest_station_value=busiest_station_value,
        # yearly totals for template
        total_2023=final_2023,
        total_2024=final_2024,
        yearly_pct_change=yearly_pct_change,
        monthly_pct_change=monthly_pct_change
    )

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('admin_panel'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('admin_login'))

@app.route('/admin')
def admin_panel():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    # Get list of existing CSV files
    csv_files = []
    if os.path.exists(UPLOAD_FOLDER):
        csv_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    
    return render_template('admin_panel.html', csv_files=csv_files)

@app.route('/admin/upload', methods=['POST'])
def admin_upload():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('admin_panel'))
    
    file = request.files['file']
    station_name = request.form.get('station_name')
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('admin_panel'))
    
    if file and allowed_file(file.filename):
        try:
            # Read the uploaded CSV
            new_data = pd.read_csv(file)
            
            # Determine target file path
            if station_name:
                target_file = os.path.join(app.config['UPLOAD_FOLDER'], f'{station_name}.csv')
            else:
                filename = secure_filename(file.filename)
                target_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # If file exists, append; otherwise create new
            if os.path.exists(target_file):
                existing_data = pd.read_csv(target_file)
                # Append new data
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                # Remove duplicates if 'date' column exists
                if 'date' in combined_data.columns:
                    combined_data = combined_data.drop_duplicates(subset=['date'], keep='last')
                combined_data.to_csv(target_file, index=False)
                flash(f'Data appended to {os.path.basename(target_file)} successfully!', 'success')
            else:
                new_data.to_csv(target_file, index=False)
                flash(f'New file {os.path.basename(target_file)} created successfully!', 'success')
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
    else:
        flash('Invalid file type. Only CSV files are allowed.', 'error')
    
    return redirect(url_for('admin_panel'))

if __name__ == '__main__':

    app.run(debug=True)
