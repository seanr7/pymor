# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Example script for solving linear PDE-constrained parameter optimization problems

Usage:
    linear_optimzation GRID_INTERVALS TRAINING_SAMPLES

Arguments:
    GRID_INTERVALS       Grid interval count.
    TRAINING_SAMPLES     Number of samples used for training the reduced model.

Options:
    -h, --help   Show this message.
"""

import numpy as np
from docopt import docopt

from pymor.basic import *


def create_fom(args):
    domain = RectDomain(([-1,-1], [1,1]))
    indicator_domain = ExpressionFunction(
        '(-2/3. <= x[..., 0]) * (x[..., 0] <= -1/3.) * (-2/3. <= x[..., 1]) * (x[..., 1] <= -1/3.) * 1. \
       + (-2/3. <= x[..., 0]) * (x[..., 0] <= -1/3.) *  (1/3. <= x[..., 1]) * (x[..., 1] <=  2/3.) * 1.',
        dim_domain=2, shape_range=())
    rest_of_domain = ConstantFunction(1, 2) - indicator_domain

    f = ExpressionFunction('0.5*pi*pi*cos(0.5*pi*x[..., 0])*cos(0.5*pi*x[..., 1])', dim_domain=2, shape_range=())

    parameters = {'diffusion': 2}
    thetas = [ExpressionParameterFunctional('1.1 + sin(diffusion[0])*diffusion[1]', parameters,
                                       derivative_expressions={'diffusion': ['cos(diffusion[0])*diffusion[1]',
                                                                             'sin(diffusion[0])']}),
          ExpressionParameterFunctional('1.1 + sin(diffusion[1])', parameters,
                                       derivative_expressions={'diffusion': ['0',
                                                                             'cos(diffusion[1])']}),

                                       ]
    diffusion = LincombFunction([rest_of_domain, indicator_domain], thetas)

    theta_J = ExpressionParameterFunctional('1 + 1/5 * diffusion[0] + 1/5 * diffusion[1]', parameters,
                                           derivative_expressions={'diffusion': ['1/5','1/5']})

    problem = StationaryProblem(domain, f, diffusion, outputs=[('l2', f * theta_J)])

    print('Discretize ...')
    mu_bar = problem.parameters.parse([np.pi/2,np.pi/2])
    fom, _ = discretize_stationary_cg(problem, diameter=1. / int(args['GRID_INTERVALS']),
                                      mu_energy_product=mu_bar)

    return fom, mu_bar

def record_results(function, parse, data, mu):
    QoI = function(mu)
    data['num_evals'] += 1
    # we need to make sure to copy the data, since the added mu will be changed inplace by minimize afterwards
    data['evaluation_points'].append([parse(mu)['diffusion'][:][0],
                                      parse(mu)['diffusion'][:][1]])
    data['evaluations'].append(QoI[0])
    print('.', end='')
    return QoI

def report(result, parse, data, reference_mu=None):
    if (result.status != 0):
        print('\n failed!')
    else:
        print('\n succeded!')
        print('  mu_min:    {}'.format(parse(result.x)))
        print('  J(mu_min): {}'.format(result.fun[0]))
        if reference_mu is not None:
            print('  absolute error w.r.t. reference solution: {:.2e}'.format(np.linalg.norm(result.x-reference_mu)))
        print('  num iterations:     {}'.format(result.nit))
        print('  num function calls: {}'.format(data['num_evals']))
        print('  time: {:.5f} seconds'.format(data['time']))
        if 'offline_time' in data:
            print('  offline time: {:.5f} seconds'.format(data['offline_time']))
    print('')

def linear_optimization_demo(args):
    fom, mu_bar = create_fom(args)

    parameter_space = fom.parameters.space(0, np.pi)
    ranges = parameter_space.ranges['diffusion']

    initial_guess = fom.parameters.parse([0.25, 0.5])

    def fom_objective_functional(mu):
        return fom.output(mu).to_numpy()
    def fom_gradient_of_functional(mu):
        return fom.output_d_mu(fom.parameters.parse(mu))

    from functools import partial
    from scipy.optimize import minimize
    from time import perf_counter

    opt_fom_minimization_data = {'num_evals': 0,
                                'evaluations' : [],
                                'evaluation_points': [],
                                'time': np.inf}
    tic = perf_counter()
    opt_fom_result = minimize(partial(record_results, fom_objective_functional, fom.parameters.parse, opt_fom_minimization_data),
                              initial_guess.to_numpy(),
                              method='L-BFGS-B',
                              jac=fom_gradient_of_functional,
                              bounds=(ranges, ranges),
                              options={'ftol': 1e-15})
    opt_fom_minimization_data['time'] = perf_counter()-tic

    reference_mu = opt_fom_result.x

    from pymor.algorithms.greedy import rb_greedy
    from pymor.reductors.coercive import CoerciveRBReductor
    from pymor.parameters.functionals import MinThetaParameterFunctional

    coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)

    training_set = parameter_space.sample_uniformly(int(args['TRAINING_SAMPLES']))
    training_set_simple = [mu['diffusion'] for mu in training_set]

    RB_reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
    RB_greedy_data = rb_greedy(fom, RB_reductor, training_set, atol=1e-2)
    rom = RB_greedy_data['rom']

    #verifying that the adjoint and sensitivity gradients are the samea
    for mu in training_set:
        gradient_with_adjoint_approach = rom.output_d_mu(mu, adjoint_approach=True)
        gradient_with_sensitivities = rom.output_d_mu(mu, adjoint_approach=False)
        np.allclose(gradient_with_adjoint_approach, gradient_with_sensitivities)

    def rom_objective_functional(mu):
        return rom.output(mu).to_numpy()
    def rom_gradient_of_functional(mu):
        return rom.output_d_mu(fom.parameters.parse(mu))

    opt_rom_minimization_data = {'num_evals': 0,
                                 'evaluations' : [],
                                 'evaluation_points': [],
                                 'time': np.inf,
                                 'offline_time': RB_greedy_data['time']}

    tic = perf_counter()
    opt_rom_result = minimize(partial(record_results, rom_objective_functional, fom.parameters.parse,
                                      opt_rom_minimization_data),
                  initial_guess.to_numpy(),
                  method='L-BFGS-B',
                  jac=rom_gradient_of_functional,
                  bounds=(ranges, ranges),
                  options={'ftol': 1e-15})
    opt_rom_minimization_data['time'] = perf_counter()-tic

    def rom_gradient_of_functional_standard_sensitivities(mu):
        return rom.output_d_mu(fom.parameters.parse(mu), adjoint_approach=False)
    opt_rom_minimization_data_sensitivities = {'num_evals': 0,
                                               'evaluations' : [],
                                               'evaluation_points': [],
                                               'time': np.inf,
                                               'offline_time': RB_greedy_data['time']}

    tic = perf_counter()
    opt_rom_result_sensitivities = minimize(partial(record_results, rom_objective_functional,
                                                    fom.parameters.parse, opt_rom_minimization_data_sensitivities),
                  initial_guess.to_numpy(),
                  method='L-BFGS-B',
                  jac=rom_gradient_of_functional_standard_sensitivities,
                  bounds=(ranges, ranges),
                  options={'ftol': 1e-15})
    opt_rom_minimization_data_sensitivities['time'] = perf_counter()-tic
    print("\nResult of optimization with FOM model and adjoint gradient")
    report(opt_fom_result, fom.parameters.parse, opt_fom_minimization_data, reference_mu)
    print("Result of optimization with ROM model and adjoint gradient")
    report(opt_rom_result, fom.parameters.parse, opt_rom_minimization_data, reference_mu)
    print("Result of optimization with ROM model but sensitivity gradient")
    report(opt_rom_result_sensitivities, fom.parameters.parse, opt_rom_minimization_data_sensitivities, reference_mu)

if __name__ == '__main__':
    args = docopt(__doc__)
    linear_optimization_demo(args)
