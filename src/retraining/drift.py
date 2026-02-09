from evidently.presets import DataDriftPreset
from evidently import Dataset, DataDefinition, Report

def data_drift(ref_df, cur_df, report_path="evidently_report.html") -> None:
    data_def = DataDefinition()

    ref_ds = Dataset.from_pandas(ref_df, data_definition=data_def)
    curr_ds = Dataset.from_pandas(cur_df, data_definition=data_def)
    drift_report = Report(metrics=[DataDriftPreset()])
    result = drift_report.run(reference_data=ref_ds, current_data=curr_ds)

    result.save_html(report_path)