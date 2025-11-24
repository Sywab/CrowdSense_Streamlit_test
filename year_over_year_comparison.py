import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_historical_comparison_plot():
    data = pd.read_csv('static/LRT_2_Annual_Ridership_2022-2024.csv')

    months = data['Month']
    ridership_2022 = data['2022'] / 1_000_000
    ridership_2023 = data['2023'] / 1_000_000
    ridership_2024 = data['2024'] / 1_000_000

    plt.figure(figsize=(12, 6))
    plt.plot(months, ridership_2022, marker='o', label='2022')
    plt.plot(months, ridership_2023, marker='s', label='2023')
    plt.plot(months, ridership_2024, marker='^', label='2024')
    plt.title('LRT-2 Monthly Ridership Comparison (2022–2024)')
    plt.xlabel('Month')
    plt.ylabel('Ridership (Millions)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/year_over_year_comparison.png')
    plt.close()