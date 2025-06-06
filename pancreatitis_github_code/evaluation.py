from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

def evaluate_model(pred_risk, time, event):
    return concordance_index(time, -pred_risk, event)

def plot_km_curve(data):
    median_risk = np.median(data['risk'])
    data['group'] = data['risk'] > median_risk

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6,4))
    for label, df_group in data.groupby('group'):
        label_str = 'High Risk' if label else 'Low Risk'
        kmf.fit(df_group['time'], event_observed=df_group['event'], label=label_str)
        kmf.plot()

    plt.title("Kaplan-Meier Survival Curve")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("km_curve.png")
    plt.show()