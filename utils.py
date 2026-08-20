import re
from sklearn.base import BaseEstimator, TransformerMixin

class SMSTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Define common Nigerian SMS regex patterns
        self.patterns = [
            # URLs and Links
            (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' __URL__ '),

            # Nigerian Phone Numbers (e.g., 080..., +234...)
            (r'(?:\+?234|0)[789][01]\d{8}', ' __PHONE__ '),

            # Nigerian Bank Account Numbers (Standard 10 digits NUBAN)
            (r'\b\d{10}\b', ' __ACCOUNT_NUM__ '),

            # Currency amounts (e.g., N50,000, 50k, 500,000 Naira)
            (r'(?:N|₦)?\d+(?:,\d+)*(?:\.\d+)?(?:k|m|b|\s*naira)?', ' __MONEY__ '),

            # USSD Codes (e.g., *737#, *131*1#)
            (r'\*\d+(?:\*\d+)*#', ' __USSD__ ')
        ]

    def fit(self, X, y=None):
        return self # Nothing to fit, just return self

    def transform(self, X, y=None):
        cleaned_X = []
        for text in X:
            text = str(text).lower()
            # Apply all regex patterns
            for pattern, replacement in self.patterns:
                text = re.sub(pattern, replacement, text)

            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()
            cleaned_X.append(text)
        return cleaned_X
