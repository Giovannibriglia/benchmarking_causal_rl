def set_learning_and_inference_objects(if_discrete: bool, approximate: bool):

    if if_discrete:
        if approximate:
            fit_method = "discrete_mlp"
            inf_method = "discrete_approx"
        else:
            fit_method = "discrete_mle"
            inf_method = "discrete_exact"
    else:
        if approximate:
            fit_method = "continuous_gaussian"
            inf_method = "continuous_gaussian"
        else:
            fit_method = "continuous_mlp_gaussian"
            inf_method = "continuous_approx"

    return fit_method, inf_method
