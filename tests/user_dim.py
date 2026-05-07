import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Parameters
num_rows = 1000

# Possible values
roles = ['Engineer', 'Manager', 'Admin', 'Analyst']
departments = ['IT', 'Sales', 'Finance', 'HR', 'R&D']
projects = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']
start_date = datetime(2005, 1, 1)
end_date = datetime(2025, 1, 1)
date_range_days = (end_date - start_date).days

# Generate data
data = []
for i in range(1, num_rows + 1):
    user_id = f"U{i:04d}"
    role = random.choice(roles)
    department = random.choice(departments)
    manager_level = random.choice([True, False])
    hire_date = start_date + timedelta(days=random.randint(0, date_range_days))
    tenure_days = (datetime.now() - hire_date).days
    project = random.choice(projects)
    
    data.append({
        'user_id': user_id,
        'role': role,
        'department': department,
        'manager_level': manager_level,
        'hire_date': hire_date.strftime('%Y-%m-%d'),
        'tenure_days': tenure_days,
        'project': project
    })

# Create DataFrame and save CSV
df = pd.DataFrame(data)
df.to_csv('user_dimensions.csv', index=False)
print("Generated user_dimensions.csv with 1000 rows.")
