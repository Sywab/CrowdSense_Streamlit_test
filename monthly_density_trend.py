import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def monthly_density_trend_plot():
    data = pd.read_csv('static/LRT_2_Annual_Ridership_2022-2024.csv')

    months = data['Month']
    ridership_2024 = data['2024'] / 1_000_000

    plt.figure(figsize=(12, 6))
    plt.plot(months, ridership_2024, marker='^', label='2024')
    plt.title('LRT-2 Monthly Ridership (2024)')
    plt.xlabel('Month')
    plt.ylabel('Ridership (Millions)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/monthly_density_trend.png')
    plt.close()