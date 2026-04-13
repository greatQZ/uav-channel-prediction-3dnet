import pandas as pd
import matplotlib.pyplot as plt
import io

# 1. Your experimental results data
data = """
Horizon,Naive Outdated,Transformer

5,637.977905,376.258330
4,548.918091,334.873169
3,466.693451,297.167450
2,379.362030,257.127625
1,286.660858,221.577347
8,898.507141,491.642761
10,1072.235962,564.436218

"""

# 2. Read the data using Pandas and sort it by Horizon
df = pd.read_csv(io.StringIO(data))
df = df.set_index('Horizon').sort_index()

# 3. Plot the results using Matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the Naive Outdated model performance
ax.plot(df.index, df['Naive Outdated'], 
        marker='x',
        linestyle='--',
        color='darkorange',
        linewidth=2,
        label='Naive Benchmark (MSE)') # <-- Changed to English

# Plot the Transformer model performance
ax.plot(df.index, df['Transformer'], 
        marker='o',
        linestyle='-',
        color='royalblue',
        linewidth=2.5,
        label='Transformer Model (MSE)') # <-- Changed to English

# 4. Set the chart titles, labels, and legend in English
ax.set_title('Model Performance comparion on different Horizons', fontsize=18, pad=20) # <-- Changed to English
ax.set_xlabel('Prediction Horizon (steps)', fontsize=14) # <-- Changed to English
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=14) # <-- Changed to English

ax.tick_params(axis='both', which='major', labelsize=12)

ax.legend(fontsize=12)

plt.tight_layout()
plt.show()