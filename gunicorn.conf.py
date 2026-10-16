import os

# TensorFlow sizes its thread pools from the host CPU count, which oversubscribes
# a small container and can stall inference. The arbiter's environment is
# inherited by every worker, so pinning it here covers the whole deployment.
# These must land before TensorFlow is imported, i.e. before the app module.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# Render injects PORT; 8080 keeps local runs working.
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# TensorFlow inference is not safe under gthread workers: a single stalled
# request holds the GIL and takes the whole worker with it, because gunicorn's
# timeout cannot interrupt a running thread. Sync workers give a clean failure
# mode — a request that exceeds `timeout` kills and replaces the worker.
worker_class = os.environ.get("GUNICORN_WORKER_CLASS", "sync")
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "1"))

# A sync worker is killed when a single request exceeds `timeout`, so this must
# stay above the slowest cold inference on a small instance.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "180"))
graceful_timeout = 30
keepalive = 5

# Keep this off: preloading imports TensorFlow in the arbiter, and forking a
# process with live TF threads is a known deadlock.
preload_app = False
