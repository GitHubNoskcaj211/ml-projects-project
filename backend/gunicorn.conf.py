def profiler_start():
    print("Starting profiler")
    import googlecloudprofiler
    googlecloudprofiler.start(service="backend")


def on_starting(server):
    print("On starting")
    profiler_start()
