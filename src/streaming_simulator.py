import time
import pandas as pd

class StreamingSimulator:
    """
    Simulate robot sensor readings arriving one record at a time.

    Each incoming record is analyzed using trained Linear Regression
    models and anomaly thresholds.

    The simulator can classify each Axis as:

    - NORMAL
    - ALERT_CANDIDATE
    - ALERT
    - ERROR_CANDIDATE
    - ERROR
    """

    def __init__(
        self,
        data,
        models,
        thresholds,
        T=4.0,
        interval=2.0
    ):
        """
        Initialize the streaming simulator.

        Parameters
        ----------
        data : pandas.DataFrame
            Testing dataset containing Time_seconds
            and Axis #1 through Axis #8.

        models : dict
            Dictionary containing one trained Linear Regression
            model for each Axis.

        thresholds : dict
            Dictionary containing MinC and MaxC for each Axis.

        T : float
            Minimum number of seconds that an abnormal condition
            must persist before becoming an ALERT or ERROR.

        interval : float
            Real waiting time between simulated readings.
        """

        # Testing data
        self.data = data

        # Trained Linear Regression models
        self.models = models

        # MinC and MaxC for every Axis
        self.thresholds = thresholds

        # Persistence threshold in seconds
        self.T = T

        # Delay between simulated records
        self.interval = interval

        # Pointer to the next record
        self.current_index = 0

        # Persistence memory----------------------------

        # Store when an Alert-level condition starts
        self.alert_start = {
            axis: None
            for axis in self.models
        }

        # Store when an Error-level condition starts
        self.error_start = {
            axis: None
            for axis in self.models
        }


    def next_record(self):
        """
        Return the next record from the testing dataset.

        Returns
        -------
        pandas.Series or None
            Next sensor record, or None if the stream has ended.
        """

        # Check if records are still available
        if self.current_index < len(self.data):

            # Read current row
            record = self.data.iloc[self.current_index]

            # Move the pointer to the next row
            self.current_index += 1

            return record

        # No more records
        return None


    def analyze_record(self, record):
        """
        Analyze one incoming sensor record.

        For every Axis:

        1. Predict the expected current.
        2. Calculate the residual.
        3. Compare residual with MinC and MaxC.
        4. Measure persistence time.
        5. Determine the current status.

        Parameters
        ----------
        record : pandas.Series
            One incoming sensor reading.

        Returns
        -------
        list
            Analysis results for all eight axes.
        """

        results = []

        # Current simulated timestamp
        current_time = float(record["Time_seconds"])

        # Linear Regression expects X to be two-dimensional.
        #
        # Example:
        #
        # [[80500.0]]
        #
        X_new = pd.DataFrame(
            [[current_time]],
            columns=["Time_seconds"]
        )

        # -----------------------------------------------------
        # Analyze every Axis independently
        # -----------------------------------------------------

        for axis in self.models:

            # Get the trained model for this Axis
            model = self.models[axis]

            # Predict expected current
            predicted = float(
                model.predict(X_new)[0]
            )

            # Get actual sensor reading
            actual = float(record[axis])

            # Calculate residual
            #
            # Residual = Actual - Predicted
            #
            residual = actual - predicted

            # Get thresholds for this specific Axis
            min_c = self.thresholds[axis]["MinC"]
            max_c = self.thresholds[axis]["MaxC"]

            # Default state
            status = "NORMAL"
            duration = 0.0

            # =================================================
            # ERROR LEVEL
            # =================================================

            if residual >= max_c:

                # We are no longer tracking an Alert sequence
                self.alert_start[axis] = None

                # Start Error timer if this is the first
                # Error-level reading
                if self.error_start[axis] is None:
                    self.error_start[axis] = current_time

                # Calculate how long the Error condition
                # has persisted
                duration = (
                    current_time
                    - self.error_start[axis]
                )

                # Confirm Error only after T seconds
                if duration >= self.T:
                    status = "ERROR"

                else:
                    status = "ERROR_CANDIDATE"

            # =================================================
            # ALERT LEVEL
            # =================================================

            elif residual >= min_c:

                # Error condition is no longer active
                self.error_start[axis] = None

                # Start Alert timer if necessary
                if self.alert_start[axis] is None:
                    self.alert_start[axis] = current_time

                # Calculate how long the Alert condition
                # has persisted
                duration = (
                    current_time
                    - self.alert_start[axis]
                )

                # Confirm Alert only after T seconds
                if duration >= self.T:
                    status = "ALERT"

                else:
                    status = "ALERT_CANDIDATE"

            # =================================================
            # NORMAL
            # =================================================

            else:

                # A normal reading interrupts any previous
                # abnormal sequence.
                self.alert_start[axis] = None
                self.error_start[axis] = None

                status = "NORMAL"
                duration = 0.0

            # -------------------------------------------------
            # Store result for this Axis
            # -------------------------------------------------

            results.append({
                "Axis": axis,
                "Actual": actual,
                "Predicted": predicted,
                "Residual": residual,
                "Duration": duration,
                "Status": status
            })

        return results


    def stream(self, number_of_records=None):
        """
        Simulate the arrival of sensor records.

        Each record is analyzed and displayed before waiting
        for the next simulated reading.

        Parameters
        ----------
        number_of_records : int or None
            Maximum number of records to process.

            If None, the complete testing dataset is streamed.
        """

        count = 0

        while True:

            # -------------------------------------------------
            # Check requested number of records
            # -------------------------------------------------

            if (
                number_of_records is not None
                and count >= number_of_records
            ):
                break

            # -------------------------------------------------
            # Get next reading
            # -------------------------------------------------

            record = self.next_record()

            # Stop when no more records are available
            if record is None:
                print("End of stream.")
                break

            # -------------------------------------------------
            # Analyze reading
            # -------------------------------------------------

            results = self.analyze_record(record)

            # Display current record
            print(
                f"\nRecord {self.current_index} | "
                f"Time: {record['Time_seconds']:.2f}"
            )

            # Display results for all eight axes
            for result in results:

                print(
                    f"{result['Axis']} | "
                    f"Actual: {result['Actual']:.3f} | "
                    f"Predicted: {result['Predicted']:.3f} | "
                    f"Residual: {result['Residual']:.3f} | "
                    f"Duration: {result['Duration']:.1f}s | "
                    f"Status: {result['Status']}"
                )

            # One record was processed
            count += 1

            # -------------------------------------------------
            # Simulate delay between sensor readings
            # -------------------------------------------------

            time.sleep(self.interval)


    def reset(self):
        """
        Reset the complete streaming simulation.

        This returns the pointer to the first record and
        clears all Alert/Error timers.
        """

        # Return pointer to first record
        self.current_index = 0

        # Clear Alert timers
        self.alert_start = {
            axis: None
            for axis in self.models
        }

        # Clear Error timers
        self.error_start = {
            axis: None
            for axis in self.models
        }