"""Banana leaf disease API package.

TensorFlow sizes its thread pools from the host CPU count, which oversubscribes
small containers and makes single-image inference far slower than the model
warrants. These must be set before TensorFlow is imported, so they live in the
package initialiser: every entry point (`app.py`, `gunicorn app:app`, tests)
imports this package before touching Keras.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
