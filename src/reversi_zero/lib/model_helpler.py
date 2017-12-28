from logging import getLogger
import requests
import tempfile
import os
import time

logger = getLogger(__name__)


def load_model_weight(model):
    return None
    retry_count_max = 10
    retry_count = 0
    while retry_count < retry_count_max:
        try:
            return load_model_weight_internal(model)
        except Exception as e:
            logger.debug(e)
            logger.debug("will retry")
            # for whatever reason(e.g., network error, fds file synchronization error), we sleep and retry.
            time.sleep(10)
            retry_count += 1

    raise Exception(f"Failed to load model after {retry_count_max} tries!")


def load_model_weight_internal(model):
    """

    :param reversi_zero.agent.model.ReversiModel model:
    :return:
    """
    cr = model.config.resource
    if cr.use_remote_model:

        config_file = tempfile.NamedTemporaryFile(delete=False)
        response = requests.get(cr.remote_model_config_path)
        config_file.write(response.content)
        config_file.close()

        weight_file = tempfile.NamedTemporaryFile(delete=False)
        response = requests.get(cr.remote_model_weight_path)
        weight_file.write(response.content)
        weight_file.close()

        logger.debug(f"using remote model from {cr.remote_model_weight_path}")
        loaded = model.load(config_file.name, weight_file.name)

        os.unlink(config_file.name)
        os.unlink(weight_file.name)

        return loaded

    else:
        return model.load(cr.model_config_path, cr.model_weight_path)


def save_model_weight(model, steps):
    """

    :param reversi_zero.agent.model.ReversiModel model:
    :return:
    """
    cr = model.config.resource
    if cr.use_remote_model:
        raise Exception("not supported yet!")  # this should be a upload
    return model.save(model.config.resource.model_config_path, model.config.resource.model_weight_path, steps)


def reload_model_weight_if_changed(model):
    """

    :param reversi_zero.agent.model.ReversiModel model:
    :return:
    """
    logger.debug(f"start reload the model if changed")

    cr = model.config.resource
    if cr.use_remote_model:
        config_file = tempfile.NamedTemporaryFile(delete=False)
        response = requests.get(cr.remote_model_config_path)
        config_file.write(response.content)
        config_file.close()

        digest = model.fetch_digest(config_file.name)

        os.unlink(config_file.name)
    else:
        digest = model.fetch_digest(cr.model_weight_path)

    if digest != model.digest:
        return load_model_weight(model)

    logger.debug(f"the model is not changed")
    return None
