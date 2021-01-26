import ray
import logging


class RayHandler:

    def __enter__(self, **kwargs):
        if ray.is_initialized():
            raise RuntimeError('Ray is already initialized')

        ray.init(**kwargs)

        logging.log(level=logging.INFO, msg='Ray successfully initialized')

    def __exit__(self, *args):
        if ray.is_initialized():
            ray.shutdown()

        logging.log(level=logging.INFO, msg='Ray successfully shutdown')
