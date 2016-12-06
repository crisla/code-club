# first line: 1
@memory.cache
def compute_value_function_cached(grid, ufunc, beta, alpha, shocks):
    """
    Compute the value function by iterating on the Bellman operator.
    The work is done by QuantEcon's compute_fixed_point function.
    """
    Tw = np.empty(len(grid))
    initial_w = 5 * np.log(grid) - 25

    v_star = compute_fixed_point(bellman_operator, 
            initial_w, 
            1e-4,  # error_tol
            100,   # max_iter
            False,  # verbose
            5,     # print_skip
            grid,
            beta,
            ufunc,
            lambda k: k**alpha,
            shocks,
            Tw=Tw,
            compute_policy=False)
    return v_star
