import googlecloudprofiler
import os


def profiler_start():
    print("Starting profiler", os.getpid())
    googlecloudprofiler.start(service="backend")


def on_starting(server):
    profiler_start()


def post_fork(server, worker):
    profiler_start()
