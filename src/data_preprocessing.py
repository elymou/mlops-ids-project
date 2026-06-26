import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Column names for NSL-KDD
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

def load_data(path):
    # ✅ Works with .txt, .csv, or any comma-separated file
    df = pd.read_csv(path, names=COLUMNS, sep=',')
    df.drop('difficulty', axis=1, inplace=True)
    return df

def preprocess(df):
    # Binary classification: normal=0, attack=1
    df['label'] = df['label'].apply(lambda x: 0 if str(x).strip() == 'normal' else 1)

    cat_cols = ['protocol_type', 'service', 'flag']

    # ✅ Save each encoder separately so API can use them later
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler and encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')

    return X_scaled, y

if __name__ == "__main__":
    # ✅ Works with .txt files directly
    df_train = load_data('data/raw/KDDTrain+.txt')
    X, y = preprocess(df_train)

    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_train.npy', X)
    np.save('data/processed/y_train.npy', y)
    print("✅ Preprocessing done!")