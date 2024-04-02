import googlecloudprofiler


def profiler_start():
    googlecloudprofiler.start(service="backend")


def on_starting(server):
    profiler_start()


def post_fork(server, worker):
    profiler_start()
