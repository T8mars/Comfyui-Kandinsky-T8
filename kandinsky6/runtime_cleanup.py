"""Always release both prefetch and patcher state without masking inference errors."""
import logging
import sys


def cleanup_sampling(model, cleanup_prefetch):
    active_error = sys.exc_info()[1]
    first_error = None
    for cleanup in (cleanup_prefetch, model.cleanup):
        try:
            cleanup()
        except Exception as error:
            logging.exception("Kandinsky 6 sampling cleanup failed")
            if first_error is None:
                first_error = error
    if first_error is not None and active_error is None:
        raise first_error
