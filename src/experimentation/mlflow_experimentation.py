import logging
import os
import uuid
from pathlib import Path

from src.experimentation import Experimentation

# Optional imports for this experimentation module
try:
    import mlflow
except ImportError:
    pass


class MlflowExperimentation(Experimentation):
    def __init__(self, tracking_uri=None):
        """
        Wrapper for the MLFlow object
        :param tracking_uri: Where runs gets stored.
        Either a local folder, or the uri of the remote (or local) tracking server.
        See https://mlflow.org/docs/0.4.0/tracking.html#where-runs-get-recorded
        """
        super().__init__()

        # if not tracking_uri:
        #    tracking_uri = self.get_local_tracking_uri()

        mlflow.set_tracking_uri(tracking_uri)

    def set_experiment(self, name, artifact_location=None):
        mlflow.set_experiment(name)

    def start_run(self):
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.start_run()

    def end_run(self):
        mlflow.end_run()

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step)

    def log_image(self, title, fig):
        # Save figure
        try:
            fig.savefig(title + ".png")
            mlflow.log_artifact(title + ".png")
        except OSError:
            # Error saving file probably due to bad title string
            unique_filename = str(uuid.uuid4()) + ".png"
            fig.savefig(unique_filename)
            mlflow.log_artifact(unique_filename)

    @staticmethod
    def search_runs(
            experiment_ids=None,
            filter_string="",
            run_view_type=1,
            max_results=100000,
            order_by=None,
    ):
        mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            run_view_type=run_view_type,
            max_results=max_results,
            order_by=order_by,
        )

    @staticmethod
    def get_local_tracking_uri():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        tracking_uri = str(Path("file:", dir_path, "../../experiments"))

        logging.info(f"Tracking uri not passed, "
                     "experiments will be logged in {tracking_uri}")

        return tracking_uri
