# module4_novel_channel_detector.py
import pandas as pd
import re
from typing import List, Dict

class NovelChannelDetector:
    """Detects data leakage through non-traditional channels"""
    
    def __init__(self):
        self.sensitive_patterns = {}
        
    def register_sensitive_patterns(self, patterns: Dict[str, str]):
        """Register regex patterns for sensitive data"""
        self.sensitive_patterns = patterns
    
    def detect_api_leakage(self, api_logs: pd.DataFrame) -> List[Dict]:
        """Scan API logs for sensitive data in prompts/requests"""
        alerts = []
        
        for idx, row in api_logs.iterrows():
            request_text = row.get('request_body', '')
            
            for pattern_name, pattern in self.sensitive_patterns.items():
                matches = re.findall(pattern, request_text)
                if matches:
                    alerts.append({
                        'user_id': row.get('user_id'),
                        'timestamp': row.get('timestamp'),
                        'pattern_matched': pattern_name,
                        'matches': matches,
                        'risk_level': 'high'
                    })
        
        return alerts
    
    def scan_screenshots(self, screenshot_dir: str) -> List[Dict]:
        """Scan screenshots for sensitive text using OCR"""
        try:
            import pytesseract
            from PIL import Image
            import os
            
            alerts = []
            for filename in os.listdir(screenshot_dir):
                if filename.endswith(('.png', '.jpg')):
                    img_path = os.path.join(screenshot_dir, filename)
                    img = Image.open(img_path)
                    text = pytesseract.image_to_string(img)
                    
                    # Check for sensitive patterns
                    for pattern_name, pattern in self.sensitive_patterns.items():
                        if re.search(pattern, text):
                            alerts.append({
                                'filename': filename,
                                'pattern_matched': pattern_name,
                                'risk_level': 'medium'
                            })
            
            return alerts
        except ImportError:
            print("Tesseract not available. Install: pip install pytesseract")
            return []

# Usage
ncd = NovelChannelDetector()
ncd.register_sensitive_patterns({
    'ssn': r'\d{3}-\d{2}-\d{4}',
    'credit_card': r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}'
})

api_alerts = ncd.detect_api_leakage(api_logs_df)
print(f"Found {len(api_alerts)} API leakage alerts")
