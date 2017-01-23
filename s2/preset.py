# prep
def ces_output(t, k, params):
    """
    Constant elasticity of substitution (CES) production function.

    Arguments:

        t:      (array) Time.
        k:      (array) Capital (per person/effective person).
        params: (dict) Dictionary of parameter values.
       
    Returns:

        y: (array-like) Output (per person/ effective person)

    """
    # extract params
    alpha = params['alpha']
    sigma = params['sigma']
    gamma = (sigma - 1) / sigma
    
    # nest Cobb-Douglas as special case
    if gamma == 0:
        y = k**alpha
    else:
        y = (alpha * k**gamma + (1 - alpha))**(1 / gamma)
    
    return y

# my solution!
def marginal_product_capital(t, k, params):
    """
    Marginal product of capital for CES production function.

    Arguments:

        t:      (array-like) Time.
        k:      (array-like) Capital (per person/effective person).
        params: (dict) Dictionary of parameter values.
       
    Returns:

        mpk: (array-like) Derivative of output with respect to capital, k.

    """
    # extract params
    alpha = params['alpha']
    sigma = params['sigma']
    gamma = (sigma - 1) / sigma
    
    # nest Cobb-Douglas as special case
    if gamma == 0:
        mpk = alpha * k**(alpha - 1)
    else:
        mpk = alpha * k**(gamma - 1) * (alpha * k**gamma + (1 - alpha))**((1 / gamma) - 1)
    
    return mpk

# my solutions!
def equation_motion_capital(t, vec, params):
    """
    Equation of motion for capital (per person/ effective person).

    Arguments:

        t:      (array-like) Time.
        vec:    (array-like) Vector of endogenous variables [k, c] where k is 
                capital (per person/effective person) and c is consumption 
                (per person/effective person).
        params: (dict) Dictionary of parameter values.
       
    Returns:

        k_dot: (array-like) Rate of change in the stock of captial 
               (per person/effective person).

    """
    # extract params
    n     = params['n']
    g     = params['g']
    delta = params['delta']
    
    # unpack the vector of endogenous variables
    k, c = vec
    
    # formula for the equation of motion for capital 
    k_dot = ces_output(t, k, params) - (n + g + delta) * k - c
    
    return k_dot

def consumption_euler_equation(t, vec, params):
    """
    Equation of motion for consumption (per person/ effective person).

    Arguments:

        t:      (array-like) Time.
        vec:    (array-like) Vector of endogenous variables [k, c] where k is 
                capital (per person/effective person) and c is consumption 
                (per person/effective person).
        params: (dict) Dictionary of parameter values.
       
    Returns:

        c_dot: (array-like) Rate of change in consumption (per person/effective 
               person).

    """
    # extract params
    g     = params['g']
    delta = params['delta']
    rho   = params['rho']
    theta = params['theta']
    
    # unpack the vector of endogenous variables
    k, c = vec
    
    # marginal product of capital
    mpk = marginal_product_capital(t, k, params)

    # formula for the equation of motion for consumption
    c_dot = ((mpk - delta - rho - theta * g) / theta) * c
    
    return c_dot

def crra_utility(c, params):
    """
    The optimal growth model assumes that the representative individual has CRRA utility.      
    utility. Individual utility is a function of consumption per effective person, c. The 
    parameter, 0 < theta, plays a dual role: 
        
        1) theta is the agent's coefficient of relative risk aversion
        2) 1 / theta is the agent's elasticity of inter-temporal substitution
    
    N.B.: If theta = 1, then CRRA utility is equivalent to log utility! 
    
    Arguments:
    
        c:      (array) Consumption per effective person
        
        params: (dict) Dictionary of model parameters.
        
    Returns: 
    
        utility: (array) An array of values respresenting utility from consumption per 
                 effective worker.
        
    """
    # extract the params
    theta = params['theta']
    
    if theta == 1:
        utility = np.log(c)
    else:
        utility = (c**(1 - theta) - 1) / (1 - theta)
    
    return utility


# declare symbolic variables
c, k = sp.symbols('c,k')

# declare symbolic parameters
n, g, alpha, delta, rho, sigma, theta = sp.symbols('n,g,alpha,delta,rho,sigma,theta')

# set sigma = 1.0 in order to specify the Cobb-Douglas special case
symbolic_params = {'n':n, 'g':g, 'alpha':alpha, 'delta':delta, 'rho':rho, 
                   'sigma':1.0, 'theta':theta}

k_dot = lambda k, c: equation_motion_capital(0, [k, c], symbolic_params)
c_dot = lambda k, c: consumption_euler_equation(0, [k, c], symbolic_params)
F     = sp.Matrix([k_dot(k, c), c_dot(k, c)])

# now computing the Jacobian 
symbolic_jacobian  = F.jacobian([k, c])

# try to simplify as much as possible
symbolic_jacobian.simplify()

def ramsey_jacobian(t, vec, params):
    """
    The Jacobian for the optimal growth model is computed using symbolic 
    differentiation and then evaluated numerically.

    Arguments:

        t:      (array-like) Time.
        vec:    (array-like) Vector of endogenous variables [k, c] where k is 
                capital (per person/effective person) and c is consumption 
                (per person/effective person).
        params: (dict) Dictionary of parameter values.
       
    Returns:

       jac: (array-like) Jacobian matrix of partial derivatives.

    """
    # declare two symbolic variables
    k, c = sp.symbols('k,c')
    
    # represent the system of equations as a SymPy matrix
    k_dot = lambda k, c: equation_motion_capital(t, [k, c], params)
    c_dot = lambda k, c: consumption_euler_equation(t, [k, c], params)
    F     = sp.Matrix([k_dot(k, c), c_dot(k, c)])

    # now computing the Jacobian is trivial!
    symbolic_jacobian  = F.jacobian([k, c])
        
    # lambdify the function in order to evaluate it numerically
    numeric_jacobian   = sp.lambdify(args=[k, c], expr=symbolic_jacobian, modules='numpy') 
    
    # output num_jac evaluated at vec
    evaluated_jacobian = np.array(numeric_jacobian(vec[0], vec[1]))
    
    return evaluated_jacobian

def analytic_k_star(params): 
    """Steady-state level of capital (per person/effective person)."""
    # extract params
    n     = params['n']
    g     = params['g']
    alpha = params['alpha']
    delta = params['delta']
    rho   = params['rho']
    theta = params['theta']
    sigma = params['sigma']
    gamma = (sigma - 1) / sigma
    
    # nest Cobb-Douglas as special case
    if gamma == 0:
        k_star = (alpha / (delta + rho + theta * g))**(1 / (1 - alpha))
    else:
        k_star = (1 - alpha)**(1 / gamma) * ((alpha / (delta + rho + theta * g))**(gamma / (gamma - 1)) - alpha)**(-1 / gamma)
    
    return k_star

def analytic_c_star(params): 
    """Steady-state level of consumption (per person/effective person)."""
    # extract params
    n     = params['n']
    g     = params['g']
    delta = params['delta']
    
    # compute k_star
    k_star = analytic_k_star(params)
    
    # compute c_star
    c_star = ces_output(0, k_star, params) - (n + g + delta) * k_star
    
    return c_star

# start by defining a dictionary of parameter values
baseline_params = {'alpha':0.33, 'n':0.025, 'g':0.025, 'rho':0.04, 'theta':2.5, 
                   'delta':0.1, 'sigma':1.0, 'A0':1.0, 'L0':1.0}

# we create an instance of the optimgal growth model as follows...
model = ramsey.Model(output=ces_output,                
                     mpk=marginal_product_capital, 
                     k_dot=equation_motion_capital, 
                     c_dot=consumption_euler_equation,
                     utility=crra_utility,
                     jacobian=ramsey_jacobian, 
                     params=baseline_params)

# create a dictionary containing the steady state expressions
steady_state_funcs = {'k_star':analytic_k_star, 'c_star':analytic_c_star}

# pass it as an arg to the set_functions method
model.steady_state.set_functions(func_dict=steady_state_funcs)

# compute the steady state values!
model.steady_state.set_values()


