# API checks
def is_valid_parameter(object):
    has_value = hasattr(object, "value")
    has_set_value = hasattr(object, "set_value")
    has_floating = hasattr(object, "floating")

    return has_value and has_set_value and has_floating


def is_valid_data(object):
    pass


def is_valid_pdf(object):
    has_params = hasattr(object, "params")
    if not has_params:
        return False
    else:
        params = object.params

    all_valid_params = all(is_valid_parameter(p) for p in params)
    has_pdf = hasattr(object, "pdf")
    has_integrate = hasattr(object, "integrate")
    has_sample = hasattr(object, "sample")
    has_space = hasattr(object, "space")

    return all_valid_params and has_pdf and has_integrate and has_sample and has_space


def is_valid_loss(object):
    has_model = hasattr(object, "model")
    if not has_model:
        return False
    else:
        model = object.model

    has_data = hasattr(object, "data")
    has_constraints = hasattr(object, "constraints")
    has_fit_range = hasattr(object, "fit_range")
    all_valid_pdfs = all(is_valid_pdf(m) for m in model)

    return all_valid_pdfs and has_data and has_constraints and has_fit_range


def is_valid_fitresult(object):
    has_loss = hasattr(object, "loss")

    if not has_loss:
        return False
    else:
        loss = object.loss
        has_params = hasattr(object, "params")
        return is_valid_loss(loss) and has_params


def is_valid_minimizer(object):
    has_minimize = hasattr(object, "minimize")
    return has_minimize
