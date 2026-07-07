import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

COLUMNS = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
'root_shell','su_attempted','num_root','num_file_creations','num_shells',
'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
'label','difficulty']

reference_df = pd.read_csv("/home/drp53/Desktop/mlops-ids-project/data/raw/KDDTrain+.txt", names=COLUMNS).drop(columns=['label', 'difficulty'])
current_df = pd.read_csv("/home/drp53/Desktop/mlops-ids-project/data/raw/KDDTest+.txt", names=COLUMNS).drop(columns=['label', 'difficulty'])

# no manual DataDefinition — let Evidently infer it
reference_dataset = Dataset.from_pandas(reference_df)
current_dataset = Dataset.from_pandas(current_df)

report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=reference_dataset, current_data=current_dataset)
result.save_html("evidently_report_v2.html")
print("Saved evidently_report_v2.html")