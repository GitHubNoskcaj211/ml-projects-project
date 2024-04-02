import googlecloudprofiler
import os


def post_fork(server, worker):
    googlecloudprofiler.start(service="backend")
