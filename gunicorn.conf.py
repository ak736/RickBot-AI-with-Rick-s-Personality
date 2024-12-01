# gunicorn.conf.py
import multiprocessing

workers = multiprocessing.cpu_count() * 2 + 1
timeout = 300  # 5 minutes
bind = "0.0.0.0:10000"