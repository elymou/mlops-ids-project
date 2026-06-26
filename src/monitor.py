import pandas as pd
import numpy as np
import os
from scipy import stats

FEATURE_COLUMNS = [
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
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

def generate_drift_report(reference_data, current_data, output_path="monitoring/drift_report.html"):
    os.makedirs("monitoring", exist_ok=True)

    results = []
    for col in reference_data.columns:
        ref_col = reference_data[col].dropna()
        cur_col = current_data[col].dropna()
        
        # KS test for drift detection
        ks_stat, p_value = stats.ks_2samp(ref_col, cur_col)
        drifted = p_value < 0.05
        
        results.append({
            "Feature": col,
            "KS Statistic": round(ks_stat, 4),
            "P-Value": round(p_value, 4),
            "Drift Detected": "🔴 YES" if drifted else "🟢 NO",
            "Ref Mean": round(ref_col.mean(), 4),
            "Cur Mean": round(cur_col.mean(), 4),
        })

    df_results = pd.DataFrame(results)
    n_drifted = df_results["Drift Detected"].str.contains("YES").sum()
    total = len(df_results)

    # Build HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Drift Report - IDS MLOps</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #0f172a; color: #e2e8f0; }}
            h1 {{ color: #38bdf8; }}
            h2 {{ color: #94a3b8; }}
            .summary {{ background: #1e293b; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
            .ok {{ color: #4ade80; font-size: 1.5em; }}
            .warn {{ color: #f87171; font-size: 1.5em; }}
            table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 10px; overflow: hidden; }}
            th {{ background: #334155; padding: 12px; text-align: left; color: #38bdf8; }}
            td {{ padding: 10px 12px; border-bottom: 1px solid #334155; }}
            tr:hover {{ background: #273449; }}
        </style>
    </head>
    <body>
        <h1>🛡️ IDS MLOps — Data Drift Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Reference samples: <strong>{len(reference_data)}</strong></p>
            <p>Current samples: <strong>{len(current_data)}</strong></p>
            <p>Features analyzed: <strong>{total}</strong></p>
            <p>Features with drift: 
                <span class="{'warn' if n_drifted > 0 else 'ok'}">
                    {n_drifted} / {total}
                </span>
            </p>
        </div>
        <h2>Feature Drift Details (KS Test)</h2>
        {df_results.to_html(index=False, border=0)}
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html)
    print(f"✅ Drift report saved to {output_path}")

if __name__ == "__main__":
    data = np.load('data/processed/X_train.npy')
    X_ref = pd.DataFrame(data[:1000], columns=FEATURE_COLUMNS)
    X_cur = pd.DataFrame(data[1000:2000], columns=FEATURE_COLUMNS)
    generate_drift_report(X_ref, X_cur)
    print("Open monitoring/drift_report.html in your browser!")