import os


def profiler_start():
    print("Starting profiler", os.getpid())
    import googlecloudprofiler
    googlecloudprofiler.start(service="backend")


def on_starting(server):
    profiler_start()
    os.register_at_fork(after_in_child=profiler_start)
