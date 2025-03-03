import asyncio
import csv
import os
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
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

    def __init__(self, exp, env, suffix):
        super().__init__(exp, env, suffix)
        self.raw_feedback: list[UnprocessedFeedback] = []
        self.feedback: list[StandardizedFeedback] = []

        self.logger_csv_path = "logs/" + self.logger_id + ".csv"
        self.raw_logger_csv_path = "logs/" + self.logger_id + "_raw.csv"

        # Service account setup
        self.credentials = service_account.Credentials.from_service_account_file(
            'service-account-key.json',
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        self.drive_service = build('drive', 'v3', credentials=self.credentials)
        self.folder_id = '18z-dZh0KMeBnSNT5g7oRWfoxYTzJ2quB'  # Your Google Drive folder ID

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
            with open(self.logger_csv_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=fb.dict().keys())
                if os.path.getsize(self.logger_csv_path) == 0:
                    writer.writeheader()
                for feedback in self.feedback:
                    writer.writerow(feedback.dict())

            try:
                # Search for existing file
                response = self.drive_service.files().list(
                    q=f"name='{self.logger_id}.csv' and '{self.folder_id}' in parents",
                    spaces='drive'
                ).execute()

                if response.get('files'):
                    # Update existing file
                    file_id = response['files'][0]['id']
                    media = MediaFileUpload(self.logger_csv_path, mimetype='text/csv')
                    self.drive_service.files().update(
                        fileId=file_id,
                        media_body=media
                    ).execute()
                else:
                    # Create new file if doesn't exist
                    file_metadata = {
                        'name': f'{self.logger_id}.csv',
                        'parents': [self.folder_id]
                    }
                    media = MediaFileUpload(self.logger_csv_path, mimetype='text/csv')
                    self.drive_service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id'
                    ).execute()

            except Exception as e:
                print(f"Drive upload error: {str(e)}")                     

            self.feedback = []

    async def dump_raw(self) -> None:
        """
        Dumps the raw feedback to the csv file
        :return: None
        """
        if len(self.raw_feedback) > 0:
            fb = self.raw_feedback[0]
            # Create dir if not exists
            with open(self.raw_logger_csv_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=fb.dict().keys())
                if os.path.getsize(self.raw_logger_csv_path) == 0:
                    writer.writeheader()
                for feedback in self.raw_feedback:
                    writer.writerow(feedback.dict())
