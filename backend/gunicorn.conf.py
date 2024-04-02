import googlecloudprofiler


def post_fork(server, worker):
    googlecloudprofiler.start(service="backend")
