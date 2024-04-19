import time


def running_time(func):
    def wrap(*args, **kwargs):
        starting_time = time.time()
        func(*args, **kwargs)
        print(f"#### Process duration is {round((time.time() - starting_time) / 60, 2)} minutes.")

    return wrap