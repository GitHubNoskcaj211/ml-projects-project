import googlecloudprofiler
import os


def post_fork(server, worker):
    print("Starting profiler", os.getpid())
    googlecloudprofiler.start(service="backend")
