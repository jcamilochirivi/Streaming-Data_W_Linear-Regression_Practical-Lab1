import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class PredictiveMaintenanceModel:
    """Reusable predictive-maintenance workflow for the Practical Lab."""

    def __init__(self, training_data, axes=None):
        self.axes = axes if axes is not None else [f"Axis #{i}" for i in range(1, 9)]
        required = ["Time_seconds"] + self.axes
        missing = [c for c in required if c not in training_data.columns]
        if missing:
            raise ValueError("Training data is missing required columns: " + ", ".join(missing))

        self.training_data = training_data.copy()
        self.models = {}
        self.model_summary = pd.DataFrame()
        self.thresholds = {}
        self.thresholds_df = pd.DataFrame()
        self.time_analysis_df = pd.DataFrame()
        self.T = None
        self.scaler = None
        self.synthetic_test_data = None
        self.metadata_comparison_df = pd.DataFrame()
        self.event_log = pd.DataFrame()

    def train_models(self):
        """Train Time_seconds -> Axis linear-regression models for all axes."""
        X = self.training_data[["Time_seconds"]]
        rows = []

        for axis in self.axes:
            y = self.training_data[axis]
            model = LinearRegression()
            model.fit(X, y)
            self.models[axis] = model

            pred_col = f"{axis}_predicted"
            res_col = f"{axis}_residual"
            self.training_data[pred_col] = model.predict(X)
            self.training_data[res_col] = self.training_data[axis] - self.training_data[pred_col]

            rows.append({
                "Axis": axis,
                "Slope": float(model.coef_[0]),
                "Intercept": float(model.intercept_),
                "R2": float(model.score(X, y)),
            })

        self.model_summary = pd.DataFrame(rows)
        return self.model_summary.copy()

    def discover_thresholds(self, min_quantile=0.95, max_quantile=0.99):
        """Derive MinC and MaxC from positive historical residuals."""
        if not self.models:
            raise RuntimeError("Models must be trained before threshold discovery.")

        rows = []
        thresholds = {}

        for axis in self.axes:
            res_col = f"{axis}_residual"
            positive = self.training_data.loc[self.training_data[res_col] > 0, res_col]
            if positive.empty:
                raise ValueError(f"No positive residuals were found for {axis}.")

            min_c = float(positive.quantile(min_quantile))
            max_c = float(positive.quantile(max_quantile))
            thresholds[axis] = {"MinC": min_c, "MaxC": max_c}
            rows.append({"Axis": axis, "MinC_P95": min_c, "MaxC_P99": max_c})

        self.thresholds = thresholds
        self.thresholds_df = pd.DataFrame(rows)
        return self.thresholds_df.copy()

    @staticmethod
    def _find_sustained_events(time_values, residual_values, threshold, max_gap=3.0):
        events = []
        start_time = None
        previous_time = None

        for current_time, residual in zip(time_values, residual_values):
            current_time = float(current_time)
            residual = float(residual)

            if residual >= threshold:
                if start_time is None:
                    start_time = current_time
                elif previous_time is not None and current_time - previous_time > max_gap:
                    events.append({
                        "Start_Time": start_time,
                        "End_Time": previous_time,
                        "Duration": previous_time - start_time,
                    })
                    start_time = current_time
                previous_time = current_time
            else:
                if start_time is not None:
                    events.append({
                        "Start_Time": start_time,
                        "End_Time": previous_time,
                        "Duration": previous_time - start_time,
                    })
                start_time = None
                previous_time = None

        if start_time is not None:
            events.append({
                "Start_Time": start_time,
                "End_Time": previous_time,
                "Duration": previous_time - start_time,
            })

        return events

    def analyze_persistence(self, candidate_times=(2, 4, 6), selected_T=4.0, max_gap=3.0):
        """Compare historical event durations and store the selected persistence T."""
        if not self.thresholds:
            raise RuntimeError("Thresholds must be discovered first.")

        rows = []
        for axis in self.axes:
            res_col = f"{axis}_residual"
            events = self._find_sustained_events(
                self.training_data["Time_seconds"].values,
                self.training_data[res_col].values,
                self.thresholds[axis]["MinC"],
                max_gap=max_gap,
            )
            durations = [event["Duration"] for event in events]
            row = {"Axis": axis, "Total_Events": len(events)}
            for candidate in candidate_times:
                row[f"Events_>=_{candidate}s"] = sum(duration >= candidate for duration in durations)
            rows.append(row)

        self.time_analysis_df = pd.DataFrame(rows)
        self.T = float(selected_T)
        return self.time_analysis_df.copy()

    def generate_synthetic_data(self, number_of_records=100, sampling_interval=2.0, random_state=42, noise_std=0.05):
        """Generate testing data from standardized historical patterns."""
        if not self.models:
            raise RuntimeError("Models must be trained before generating test data.")

        rng = np.random.default_rng(random_state)
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(self.training_data[self.axes])
        scaled_df = pd.DataFrame(scaled, columns=self.axes)

        indices = rng.choice(len(scaled_df), size=number_of_records, replace=True)
        synthetic_scaled = scaled_df.iloc[indices].to_numpy().copy()
        synthetic_scaled += rng.normal(0.0, noise_std, size=synthetic_scaled.shape)
        synthetic_values = self.scaler.inverse_transform(synthetic_scaled)

        synthetic = pd.DataFrame(synthetic_values, columns=self.axes)
        for axis in self.axes:
            synthetic[axis] = synthetic[axis].clip(
                lower=self.training_data[axis].min(),
                upper=self.training_data[axis].max(),
            )

        last_time = float(self.training_data["Time_seconds"].iloc[-1])
        synthetic.insert(
            0,
            "Time_seconds",
            last_time + np.arange(1, number_of_records + 1) * sampling_interval,
        )

        X_test = synthetic[["Time_seconds"]]
        for axis in self.axes:
            pred_col = f"{axis}_predicted"
            res_col = f"{axis}_residual"
            synthetic[pred_col] = self.models[axis].predict(X_test)
            synthetic[res_col] = synthetic[axis] - synthetic[pred_col]

        comparison = []
        for axis in self.axes:
            comparison.append({
                "Axis": axis,
                "Training_Mean": self.training_data[axis].mean(),
                "Synthetic_Mean": synthetic[axis].mean(),
                "Training_Std": self.training_data[axis].std(),
                "Synthetic_Std": synthetic[axis].std(),
            })

        self.synthetic_test_data = synthetic
        self.metadata_comparison_df = pd.DataFrame(comparison)
        return self.synthetic_test_data.copy(), self.metadata_comparison_df.copy()

    def inject_controlled_anomalies(self, axis="Axis #1", alert_rows=range(20, 24), error_rows=range(60, 64), error_multiplier=1.20):
        """Inject known sustained Alert and Error scenarios for validation."""
        if self.synthetic_test_data is None:
            raise RuntimeError("Synthetic testing data must be generated first.")
        if not self.thresholds:
            raise RuntimeError("Thresholds must be discovered first.")
        if axis not in self.axes:
            raise ValueError(f"Unknown axis: {axis}")

        min_c = self.thresholds[axis]["MinC"]
        max_c = self.thresholds[axis]["MaxC"]
        pred_col = f"{axis}_predicted"
        res_col = f"{axis}_residual"

        alert_residual = (min_c + max_c) / 2
        error_residual = max_c * error_multiplier

        for row in list(alert_rows):
            self.synthetic_test_data.loc[row, axis] = self.synthetic_test_data.loc[row, pred_col] + alert_residual
        for row in list(error_rows):
            self.synthetic_test_data.loc[row, axis] = self.synthetic_test_data.loc[row, pred_col] + error_residual

        self.synthetic_test_data[res_col] = self.synthetic_test_data[axis] - self.synthetic_test_data[pred_col]
        return self.synthetic_test_data.copy()

    @staticmethod
    def _create_axis_event_log(df, axis, min_c, max_c, T):
        res_col = f"{axis}_residual"
        events = []
        event_type = None
        event_start = None
        detection_time = None
        previous_time = None

        for _, row in df.iterrows():
            current_time = float(row["Time_seconds"])
            residual = float(row[res_col])

            if residual >= max_c:
                current_type = "ERROR"
            elif residual >= min_c:
                current_type = "ALERT"
            else:
                current_type = "NORMAL"

            if current_type == "NORMAL":
                if event_type is not None:
                    duration = previous_time - event_start
                    if duration >= T:
                        events.append({
                            "Axis": axis,
                            "Event": event_type,
                            "Start_Time": event_start,
                            "Detection_Time": detection_time,
                            "End_Time": previous_time,
                            "Duration": duration,
                        })
                event_type = None
                event_start = None
                detection_time = None
            else:
                if event_type != current_type:
                    if event_type is not None:
                        duration = previous_time - event_start
                        if duration >= T:
                            events.append({
                                "Axis": axis,
                                "Event": event_type,
                                "Start_Time": event_start,
                                "Detection_Time": detection_time,
                                "End_Time": previous_time,
                                "Duration": duration,
                            })
                    event_type = current_type
                    event_start = current_time
                    detection_time = None

                duration = current_time - event_start
                if duration >= T and detection_time is None:
                    detection_time = current_time

            previous_time = current_time

        if event_type is not None:
            duration = previous_time - event_start
            if duration >= T:
                events.append({
                    "Axis": axis,
                    "Event": event_type,
                    "Start_Time": event_start,
                    "Detection_Time": detection_time,
                    "End_Time": previous_time,
                    "Duration": duration,
                })

        return events

    def detect_events(self, testing_data=None, T=None):
        """Detect sustained Alert/Error events across all configured axes."""
        if testing_data is None:
            if self.synthetic_test_data is None:
                raise RuntimeError("No testing data is available.")
            testing_data = self.synthetic_test_data
        if not self.thresholds:
            raise RuntimeError("Thresholds must be discovered first.")

        persistence_T = float(T) if T is not None else self.T
        if persistence_T is None:
            raise RuntimeError("Persistence threshold T has not been selected.")

        events = []
        for axis in self.axes:
            events.extend(self._create_axis_event_log(
                testing_data,
                axis,
                self.thresholds[axis]["MinC"],
                self.thresholds[axis]["MaxC"],
                persistence_T,
            ))

        self.event_log = pd.DataFrame(events, columns=[
            "Axis", "Event", "Start_Time", "Detection_Time", "End_Time", "Duration"
        ])
        return self.event_log.copy()

    def get_positive_residuals(self, axis):
        """Return positive historical residuals for one axis."""
        if axis not in self.axes:
            raise ValueError(f"Unknown axis: {axis}")
        res_col = f"{axis}_residual"
        if res_col not in self.training_data.columns:
            raise RuntimeError("Models must be trained first.")
        return self.training_data.loc[self.training_data[res_col] > 0, res_col].copy()
