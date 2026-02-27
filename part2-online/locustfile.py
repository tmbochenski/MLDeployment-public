import os

from locust import HttpUser, task, between
from google.auth import default
from google.auth.transport.requests import Request


WAIT_MIN = float(os.getenv("WAIT_MIN", "0.1"))
WAIT_MAX = float(os.getenv("WAIT_MAX", "0.5"))


def get_gcloud_identity_token() -> str:
    creds, project = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    creds.refresh(Request())
    return creds.token


# todo: implement locust user

