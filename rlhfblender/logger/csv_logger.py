import asyncio
import csv
import os
from io import StringIO
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, storage
import time

from pydantic import BaseModel

from rlhfblender.data_models import StandardizedFeedback, UnprocessedFeedback

from .logger import Logger

background_tasks = set()


class CSVLogger(Logger):
    """
    This class implements a logger that logs feedback to a csv file.

    :param exp: The experiment object
    :param env: The environment object
    :param suffix: The suffix for the logger ID
    """

    def __init__(self, exp, env, suffix, storage_type="firebase"):
        super().__init__(exp, env, suffix)

        self.raw_feedback: list[UnprocessedFeedback] = []
        self.feedback: list[StandardizedFeedback] = []

        self.storage_type = storage_type

        # Local file names
        self.logger_csv_path = "logs/" + self.logger_id + ".csv"
        self.raw_logger_csv_path = "logs/" + self.logger_id + "_raw.csv"

        # Firebase file names
        self.logger_csv_name = f"{self.logger_id}.csv"
        self.raw_logger_csv_name = f"{self.logger_id}_raw.csv"

        # Initialize Firebase and get bucket directly
        if storage_type in ["firebase", "both"]:
            unique_app_name = f'rlhf-storage-{int(time.time())}'
            cred = credentials.Certificate("firebase-credentials.json")
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'rlhf-blender.firebasestorage.app'
            }, name=unique_app_name)
            
            self.bucket = storage.bucket(app=firebase_admin.get_app(name=unique_app_name))
        else:
            self.bucket = None

    def reset(self) -> None:
        """
        Resets the logger
        :return: None
        """
        super().reset()
        self.logger_csv_path = "logs/" + self.logger_id + ".csv"
        self.raw_logger_csv_path = "logs/" + self.logger_id + "_raw.csv"

    def log(self, feedback):
        """
        Logs standardized feedback
        :param feedback: The feedback
        :return: None
        """
        self.feedback.append(feedback)
        _task = asyncio.create_task(self.dump())
        background_tasks.add(_task)

    def read(self) -> list[StandardizedFeedback]:
        """
        Reads the feedback from the logger
        :return: The feedback
        """
        return self.feedback

    def log_raw(self, feedback: UnprocessedFeedback) -> None:
        """
        Logs raw feedback
        :param feedback: The feedback
        :return: None
        """
        self.raw_feedback.append(feedback)
        _task = asyncio.create_task(self.dump_raw())
        background_tasks.add(_task)

    def read_raw(self) -> list[UnprocessedFeedback]:
        """
        Reads the raw feedback from the logger
        :return: The raw feedback
        """
        return self.raw_feedback

    async def dump(self) -> None:
        """
        Dumps the processed feedback to the csv file
        :return: None
        """
        # Append the feedback to the list in the csv file: Translate feedback to csv format
        if len(self.feedback) > 0:
            fb: BaseModel = self.feedback[0]

            metadata = {
            'experiment_id': self.exp.id,            
            'timestamp': str(datetime.now()),
            'session_id': self.logger_id,
            'feedback_type': 'processed',
            }

            if self.storage_type in ["local", "both"]:
                os.makedirs(os.path.dirname(self.logger_csv_path), exist_ok=True)
                with open(self.logger_csv_path, "a") as f:
                    writer = csv.DictWriter(f, fieldnames=fb.dict().keys())
                    if os.path.getsize(self.logger_csv_path) == 0:
                        writer.writeheader()
                    for feedback in self.feedback:
                        writer.writerow(feedback.dict())

            # Handle Firebase storage
            if self.storage_type in ["firebase", "both"]:
                csv_content = StringIO()
                writer = csv.DictWriter(csv_content, fieldnames=fb.dict().keys())
                writer.writeheader()  # Write header for Firebase Storage version
                for feedback in self.feedback:
                    writer.writerow(feedback.dict())

                # Upload to Firebase Storage
                blob = self.bucket.blob(self.logger_csv_name)
                blob.metadata = metadata
                blob.upload_from_string(csv_content.getvalue(), content_type="text/csv")

        self.feedback = []

    async def dump_raw(self) -> None:
        """
        Dumps the raw feedback to the csv file
        :return: None
        """
        if len(self.raw_feedback) > 0:
            fb = self.raw_feedback[0]

            metadata = {
            'experiment_id': self.exp.id,            
            'timestamp': str(datetime.now()),
            'session_id': self.logger_id,
            'feedback_type': 'processed',
            }
            # Create dir if not exists
            if self.storage_type in ["local", "both"]:
                os.makedirs(os.path.dirname(self.raw_logger_csv_path), exist_ok=True)
                with open(self.raw_logger_csv_path, "a") as f:
                    writer = csv.DictWriter(f, fieldnames=fb.dict().keys())
                    if os.path.getsize(self.raw_logger_csv_path) == 0:
                        writer.writeheader()
                    for feedback in self.raw_feedback:
                        writer.writerow(feedback.dict())

            # Handle Firebase storage
            if self.storage_type in ["firebase", "both"]:
                csv_content = StringIO()
                writer = csv.DictWriter(csv_content, fieldnames=fb.dict().keys())
                writer.writeheader()  # Write header for Firebase Storage version
                for feedback in self.raw_feedback:
                    writer.writerow(feedback.dict())

                # Upload to Firebase Storage
                blob = self.bucket.blob(self.raw_logger_csv_name)
                blob.metadata = metadata
                blob.upload_from_string(csv_content.getvalue(), content_type="text/csv")
            
            # clear the list after processing
            self.raw_feedback = []
