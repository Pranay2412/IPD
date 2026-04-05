import pandas as pd
import numpy as np
import os

# Step 1: Create synthetic users
users = [f"U{str(i).zfill(3)}" for i in range(1, 1001)]  # Adjust number as per CERT data

# Step 2: Build synthetic attributes
user_dims = pd.DataFrame({
    'user_id': users,
    'role': np.random.choice(['Engineer', 'Manager', 'Analyst', 'Admin'], size=len(users)),
    'department': np.random.choice(['IT', 'Finance', 'R&D', 'Sales', 'HR'], size=len(users)),
    'manager_level': np.random.choice([True, False], size=len(users), p=[0.2, 0.8]),
    'hire_date': pd.to_datetime(np.random.choice(
        pd.date_range('2005-01-01', '2020-12-31'), size=len(users)
    ))
})

# Step 3: Compute derived feature (tenure)
current_date = pd.to_datetime("2025-07-21")
user_dims['tenure_days'] = (current_date - user_dims['hire_date']).dt.days

# os.makedirs("../data", exist_ok=True)

# Step 4: Save as CSV
target_path = os.path.abspath(os.path.join(__file__, "../../../data/user_dimensions.csv"))
os.makedirs(os.path.dirname(target_path), exist_ok=True)

# Save the file
user_dims.to_csv(target_path, index=False)
print(f"✅ Synthetic user_dimensions.csv saved to {target_path}")