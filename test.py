import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Possible values for categorical columns
sources = ['direct call', 'facebook', 'whatsaap', 'instagram', 'no source']
sales_managers = ['tabarak', 'basmala', 'islam', 'ibrahiem', 'ziad', 'samar']
interests = ['java', 'c', 'php', 'sql', 'R', 'python']

# Generate random data
num_rows = 100
data = {
    'index': np.arange(num_rows),
    'source': np.random.choice(sources, num_rows),
    'sales_manager': np.random.choice(sales_managers, num_rows),
    'interest': np.random.choice(interests, num_rows),
    'probability': np.random.uniform(0, 100, num_rows),
    'reservation_success': np.random.choice(['yes', 'no'], num_rows)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("reservation_data.csv", index=False)

print("Dataset with 1000 rows created and saved to 'reservation_data.csv'.")
