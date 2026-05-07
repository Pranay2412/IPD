import pandas as pd
from datetime import datetime, timedelta
import random

# Generate synthetic API logs
def generate_synthetic_api_logs(num_entries=100):
    users = ['U001', 'U002', 'U003', 'U004', 'U005']
    endpoints = ['LLM-Prompt-API', 'HR-Data-API', 'Finance-Query-API', 'Cloud-Storage-API']
    sensitive_data_samples = ['customer ID 123456', 'salary 78910', 'SSN 987-65-4321', 'credit card 4111-1111-1111-1111']

    api_logs = []
    base_time = datetime.now()

    for i in range(num_entries):
        user_id = random.choice(users)
        endpoint = random.choice(endpoints)
        timestamp = base_time - timedelta(minutes=random.randint(0, 1000))
        
        # Randomly insert sensitive data or benign content
        if random.random() < 0.3:
            request_body = f"Request to fetch {random.choice(sensitive_data_samples)}"
        else:
            request_body = "Regular API call with non-sensitive data"

        api_logs.append({
            'user_id': user_id,
            'timestamp': timestamp.isoformat(),
            'request_body': request_body,
            'endpoint': endpoint
        })

    return pd.DataFrame(api_logs)

# Usage
api_logs_df = generate_synthetic_api_logs(50)
print(api_logs_df.head())

