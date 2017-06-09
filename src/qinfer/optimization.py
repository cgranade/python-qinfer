#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_models.py: Simple models for testing inference engines.
##
# © 2017 Alan Robertson (arob8086@uni.sydney.edu.au)
#
# This file is a part of the Qinfer project.
# Licensed under the AGPL version 3.
##
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

## FEATURES ##################################################################

from __future__ import division, absolute_import, print_function

## EXPORTS ###################################################################

__all__ = [
    'ParticleSwarmOptimizer',
    'ParticleSwarmSimpleAnnealingOptimizer',
    'ParticleSwarmTemperingOptimizer',
    'SPSATwoSiteOptimizer',
    'Fitness',
    'HeuristicPerformanceFitness'
]

## IMPORTS ###################################################################

import random
import numpy as np
from functools import partial

from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass

from qinfer.perf_testing import perf_test_multiple, apply_serial, timing
from qinfer.distributions import UniformDistribution

## CLASSES ####################################################################

class Optimizer(with_metaclass(ABCMeta, object)):
    '''
    A generic optimizer class that is inherited by the other optimisation functions.

    :param np.ndarray param_names: The list of parameters that are being searched over.
    :param fitness_function: The fitness function to be optimized. If no function is
        specified, defaults to optimizing heuristic performance using
        :class:`~qinfer.HeuristicPerformanceFitness`.
    :type fitness_function: `callable`, :func:`~qinfer.Fitness`, or `partial`.
    :param callable projection_fn: Function to project points onto feasible solutions.
        If `None`, no projection is performed and the optimization problem is taken to
        be unconstrained.
    :param dict funct_args: Arguments to pass to the fitness function.
    :param dict funct_kwargs: Keyword arguments to pass to the fitness function.
    '''

    def __init__(self,
                 param_names, fitness_function=None, projection_fn=None,
                 quiet=True,
                 *funct_args, **funct_kwargs):

        self._param_names = param_names
        self._n_free_params = len(param_names)
        self._projection_fn = projection_fn
        self._funct_args = funct_args
        self._funct_kwargs = funct_kwargs
        self._quiet = bool(quiet)

        if fitness_function is None: # Default to calling perf test multiple
            self._fitness_function = HeuristicPerformanceFitness(
                self._param_names,
                *self._funct_args,
                **self._funct_kwargs
            )
        elif funct_args or funct_kwargs:
            # Partially apply any arguments that we're given.
            self._fitness_function = partial(
                fitness_function, *self._funct_args, **self._funct_kwargs
            )
        else:
            # Assume that the fitness function has already been constructed
            # and is directly callable.
            self._fitness_function = fitness_function

    def _report_iteration(self, idx_iter, fitnesses, elapsed_time=None):
        print("{: 3d}\t{:+1.3e}\t{}".format(
            idx_iter, np.max(fitnesses),
            elapsed_time if elapsed_time else ""
        ))

    # Member function needed for parallelization.
    def fitness(self, params):
        return self._fitness_function(params)

    def parallel(self):
        raise NotImplementedError("This optimizer does not have parallel support.")

    def project(self, params):
        if self._projection_fn:
            return self._projection_fn(params)
        else:
            return params

    def evaluate_fitness(self, particles, apply=apply_serial):
        # Use partial to wrap our fitness method; this makes it easier if
        # the apply implementation we're given uses pickle to communicate with
        # remote engines, as then the metadata in functools.partial will store a
        # pickleable-reference to self.
        fitness_function = partial(self.fitness)
        # We then use the given apply implementation to asynchronously evaluate
        # each fitness, then wait on the async results.
        results = [apply(fitness_function, particle) for particle in particles]

        return [result.get() for result in results]

    def update_positions(self, positions, velocities):
        return positions + velocities

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class ParticleSwarmOptimizer(Optimizer):
    '''
        A particle swarm optimisation based hyperheuristic
        :param integer n_pso_iterations:
        :param integer n_pso_particles:
        :param QInferDistribution initial_velocity_distribution:
        :param QInferDistribution initial_velocity_distribution:
        :param double 
        :param double
        :param function
    '''

    def _initialize_particle_swarm(
            self,
            n_pso_iterations, n_pso_particles,
            initial_position_distribution,
            initial_velocity_distribution,
            omega_v, phi_p, phi_g,
            apply
        ):

        with timing() as elapsed_time:
            particles = np.empty([n_pso_iterations, n_pso_particles], dtype=self.particles_dt())

            # Initialize positions, defaulting to the [0, 1] hypercube.
            if initial_position_distribution is None:
                initial_position_distribution = UniformDistribution(
                    np.array([[0, 1]] * self._n_free_params)
                )
            particles[0]["params"] = initial_position_distribution.sample(n_pso_particles)

            # Find the fitnesses of the initial particles.
            particles[0]["fitness"] = self.evaluate_fitness(
                particles[0]["params"], apply=apply
            )

            # Project the initial particles to the feasible space as appropriate.
            particles[0]["params"] = self.project(particles[0]["params"])

            # Initialize velocities, defaulting to the [-1, 1] hypercube.
            if initial_velocity_distribution is None:
                initial_velocity_distribution = UniformDistribution(
                    np.array([[-1, 1]] * self._n_free_params)
                )
            particles[0]["velocities"] = initial_velocity_distribution.sample(n_pso_particles)

            # Initialize the attractors.
            local_attractors, global_attractor = self.update_attractors(
                particles[0]
            )

            # Use the initialized attractors to set the initial velocities.
            particles[0]["velocities"] = self.update_velocities(
                particles[0]["params"],
                particles[0]["velocities"],
                local_attractors["params"],
                global_attractor["params"],
                omega_v, phi_p, phi_g
            )
        
        # Report the first iteration.
        self._report_iteration(0, particles[0]['fitness'], elapsed_time)

        return particles, local_attractors, global_attractor

    def _pso_iteration(
            self, idx_iter, previous_particles, current_particles,
            local_attractors,
            omega_v, phi_p, phi_g,
            apply
    ):
        with timing() as elapsed_time:
            # Update the particle positions using their current velocities.
            current_particles["params"] = self.update_positions(
                previous_particles["params"],
                previous_particles["velocities"]
            )

            # Project back any particles which left the space of feasible solutions.
            current_particles["params"] = self.project(current_particles["params"])

            # Recalculate the fitness function at the new particle positions.
            current_particles["fitness"] = self.evaluate_fitness(
                current_particles["params"],
                apply=apply
            )

            # Find the new attractors.
            local_attractors, global_attractor = self.update_attractors(
                current_particles,
                local_attractors
            )

            # Update the velocities
            current_particles["velocities"] = self.update_velocities(
                current_particles["params"],
                previous_particles["velocities"],
                local_attractors["params"],
                global_attractor["params"],
                omega_v, phi_p, phi_g
            )

        self._report_iteration(idx_iter, current_particles["fitness"], elapsed_time)

        return local_attractors, global_attractor

    def __call__(
            self,
            n_pso_iterations=50,
            n_pso_particles=60,
            initial_position_distribution=None,
            initial_velocity_distribution=None,
            omega_v=0.35,
            phi_p=0.25,
            phi_g=0.5,
            apply=apply_serial,
            return_all=False
    ):

        particles, local_attractors, global_attractor = self._initialize_particle_swarm(
            n_pso_iterations, n_pso_particles,
            initial_position_distribution, initial_velocity_distribution,
            omega_v, phi_p, phi_g,
            apply=apply
        )

        for idx_iter in range(1, n_pso_iterations):
            previous_particles, current_particles = particles[idx_iter - 1:idx_iter + 1]
            local_attractors, global_attractor = self._pso_iteration(
                idx_iter,
                previous_particles, current_particles,
                local_attractors,
                omega_v, phi_p, phi_g,
                apply
            )

        if return_all:
            return global_attractor, {
                'particles': particles
            }
        else:
            return global_attractor

    def update_velocities(self, positions, velocities, local_attractors, global_attractor, omega_v, phi_p, phi_g):
        random_p = np.random.random_sample(positions.shape)
        random_g = np.random.random_sample(positions.shape)
        updated = omega_v * velocities + phi_p * random_p * (local_attractors - positions) + phi_g * random_g * (global_attractor - positions) 
        return updated

    def update_attractors(self, particles, local_attractors=None):
        # If we don't yet have attractors, then we're using update_attractors
        # to initialize the PSO, so let's pick them to be the current particle
        # locations.
        if local_attractors is None:
            local_attractors = particles.copy()

        # If we do have attractors, though, then we need to check if our
        # current particles do better than them, and update accordingly.
        else:
            for idx, particle in enumerate(particles):
                # If the particle under consideration is better (lower fitness)
                # than its personal best (the corresponding local attractor),
                # then we need to update that local attractor.
                if particle["fitness"] < local_attractors[idx]["fitness"]:
                    local_attractors[idx] = particle

        # In either case, the new global attractor should be the best (smallest) fitness
        # we've seen so far.
        global_attractor = local_attractors[np.argmin(local_attractors["fitness"])].copy()

        return local_attractors, global_attractor

    def particles_dt(self):
        return np.dtype([
            ('params', np.float64, (self._n_free_params,)),
            ('velocities', np.float64, (self._n_free_params,)),
            ('fitness', np.float64)])


class ParticleSwarmSimpleAnnealingOptimizer(ParticleSwarmOptimizer):

    def __call__(
            self,
            n_pso_iterations=50,
            n_pso_particles=60,
            initial_position_distribution=None,
            initial_velocity_distribution=None,
            omega_v=0.35,
            phi_p=0.25,
            phi_g=0.5,
            temperature=0.95,
            apply=apply_serial,
            return_all=False
    ):

        particles, local_attractors, global_attractor = self._initialize_particle_swarm(
            n_pso_iterations, n_pso_particles,
            initial_position_distribution, initial_velocity_distribution,
            omega_v, phi_p, phi_g,
            apply=apply
        )

        for idx_iter in range(1, n_pso_iterations):
            previous_particles, current_particles = particles[idx_iter - 1:idx_iter + 1]
            local_attractors, global_attractor = self._pso_iteration(
                idx_iter,
                previous_particles, current_particles,
                local_attractors,
                omega_v, phi_p, phi_g,
                apply
            )

            # Update the PSO params by using a simple annealing rule.
            omega_v, phi_p, phi_g = self.update_pso_params(
                temperature,
                omega_v,
                phi_p,
                phi_g
            )

        if return_all:
            return global_attractor, {
                'particles': particles
            }
        else:
            return global_attractor

    def update_pso_params(self, temperature, omega_v, phi_p, phi_g):
        omega_v, phi_p, phi_g = np.multiply(temperature, [omega_v, phi_p, phi_g])
        return omega_v, phi_p, phi_g


class ParticleSwarmTemperingOptimizer(ParticleSwarmOptimizer):
    '''
        A particle swarm optimisation based hyperheuristic
        :param integer n_pso_iterations:
        :param integer n_pso_particles:
        :param QInferDistribution initial_velocity_distribution:
        :param QInferDistribution initial_velocity_distribution:
        :param double 
        :param double
        :param function
    '''
    
    def __call__(self,
        n_pso_iterations=50,
        n_pso_particles=60,
        initial_position_distribution=None,
        initial_velocity_distribution=None,
        n_temper_categories = 6,
        temper_frequency = 10,
        temper_params = None,
        apply=apply_serial
        ):
        self._fitness = np.empty([n_pso_iterations, n_pso_particles], dtype=self.particles_dt())
        local_attractors = np.empty([n_pso_particles], dtype=self.particles_dt())
        global_attractor = np.empty([1], dtype=self.particles_dt())

        if initial_position_distribution is None:
            initial_position_distribution = UniformDistribution(np.array([[ 0, 1]] * self._n_free_params));
            
        if initial_velocity_distribution is None:
            initial_velocity_distribution = UniformDistribution(np.array([[-1, 1]] * self._n_free_params))

        if temper_params is None:
            omega_v = np.random.random(n_temper_categories)
            phi_p = np.random.random(n_temper_categories)
            phi_g = np.random.random(n_temper_categories)
            temper_params = [np.array((params), dtype=self.temper_params_dt()) for params in zip(omega_v, phi_p, phi_g)]

        # Distribute the particles into different temper categories
        temper_map = self.distribute_particles(n_pso_particles, n_temper_categories)

        # Initial particle positions
        self._fitness[0]["params"] = initial_position_distribution.sample(n_pso_particles)
            
        # Apply the boundary conditions if any exist
        if self._projection_fn is not None:
            self._fitness[0]["params"] = self._projection_fn(self._fitness[0]["params"])

        # Calculate the initial particle fitnesses
        self._fitness[0]["fitness"] = self.evaluate_fitness(self._fitness[0]["params"], 
                                                            apply=apply)

        # Calculate the positions of the attractors
        local_attractors = self._fitness[0]
        local_attractors, global_attractor = self.update_attractors(
                                                self._fitness[0], 
                                                local_attractors, 
                                                global_attractor)

        # Initial particle velocities
        self._fitness[0]["velocities"] = initial_velocity_distribution.sample(n_pso_particles)

        # Update the velocities using the temper map
        for idx, temper_category in enumerate(temper_map):
            self._fitness[0]["velocities"][temper_category] = self.update_velocities(
                                                self._fitness[0]["params"][temper_category], 
                                                self._fitness[0]["velocities"][temper_category], 
                                                local_attractors["params"][temper_category],
                                                global_attractor["params"],
                                                temper_params[idx]["omega_v"],
                                                temper_params[idx]["phi_p"],
                                                temper_params[idx]["phi_g"])

        for itr in range(1, n_pso_iterations):
            # Update the particle positions
            self._fitness[itr]["params"] = self.update_positions(
                self._fitness[itr - 1]["params"], 
                self._fitness[itr - 1]["velocities"])

            # Apply the boundary conditions if any exist
            if self._projection_fn is not None:
                self._fitness[itr]["params"] = self._projection_fn(self._fitness[itr]["params"])

            # Recalculate the fitness function
            self._fitness[itr]["fitness"] = self.evaluate_fitness(
                self._fitness[itr]["params"],
                apply=apply)

            # Find the new attractors
            local_attractors, global_attractor = self.update_attractors(
                self._fitness[itr], 
                local_attractors, 
                global_attractor)

            # Update the velocities
            # FIXME: idx isn't actually used here, except to effectively associate
            #        temper_map and temper_params. Should be able to zip.
            for idx, temper_category in enumerate(temper_map):
                self._fitness[itr]["velocities"][temper_category] = self.update_velocities(
                                                    self._fitness[itr]["params"][temper_category], 
                                                    self._fitness[itr - 1]["velocities"][temper_category], 
                                                    local_attractors["params"][temper_category],
                                                    global_attractor["params"],
                                                    temper_params[idx]["omega_v"],
                                                    temper_params[idx]["phi_p"],
                                                    temper_params[idx]["phi_g"])
            
            # Redistribute the particles into different temper categories
            if itr % temper_frequency == 0:
                temper_map = self.distribute_particles(n_pso_particles, n_temper_categories)
                
        return global_attractor
    
    def temper_params_dt(self):
            return np.dtype([
            ('omega_v', np.float64),
            ('phi_p', np.float64),
            ('phi_g', np.float64)])


    def distribute_particles(self, n_pso_particles, n_temper_categories):

        # Distribute as many particles as evenly as possible across the categories, 
        # This ensures that there are no empty categories
        n_evenly_distributable = (n_pso_particles // n_temper_categories) * n_temper_categories
        n_unevenly_distributable = n_pso_particles - n_evenly_distributable

        # Generate the required indicies for the pso particles
        particle_indicies = list(range(0, n_pso_particles))

        # Randomise the order
        np.random.shuffle(particle_indicies)

        # Reshape to a 2D array indexed on the number of tempering categories
        particle_map = np.reshape(
            particle_indicies[:n_evenly_distributable], 
            (n_temper_categories, n_evenly_distributable//n_temper_categories))

        # Transfer to the map
        temper_map = {} 
        for i, index_category in enumerate(particle_map):
            temper_map[i] = index_category

        # Transfer any excess particles that could not be evenly distributed
        # This is a slow operation, so for the purposes of speed the number of 
        # temper categories should be a factor of the number of pso particles
        if n_unevenly_distributable != 0:
            for i in range(n_evenly_distributable, n_pso_particles):
                temper_map[random.randrange(0, n_temper_categories)] = (
                    np.append(temper_map[random.randrange(0, n_temper_categories)], [particle_indicies[i]]))

        return temper_map


class SPSATwoSiteOptimizer(Optimizer):

    def __call__(self,
        n_spsa_iterations = 60,
        n_spsa_particles = 50,
        initial_position_distribution = None,
        A = 0,
        s = 1/3,
        t = 1,
        a = 0.5,
        b = 0.5,
        apply=apply_serial
        ):

        self._fitness = np.empty([n_spsa_iterations, n_spsa_particles], dtype=self.particles_dt())

        if initial_position_distribution is None:
            initial_position_distribution = UniformDistribution(np.array([[0, 1]] * self._n_free_params));
              
        # Initial particle positions
        self._fitness[0]["params"] = initial_position_distribution.sample(n_spsa_particles)
            
        # Apply the boundary conditions if any exist
        if self._projection_fn is not None:
            self._fitness[0]["params"] = self._projection_fn(self._fitness[0]["params"])

        # Calculate the initial particle fitnesses
        self._fitness[0]["fitness"] = self.evaluate_fitness(self._fitness[0]["params"], 
                                                            apply=apply)

        for itr in range(1, n_spsa_iterations):

            # Helper functions to determine the update
            delta_k = self.delta(n_spsa_particles, self._n_free_params)
            first_site = np.vstack(
                self.evaluate_fitness(
                    self._fitness[itr-1]["params"] - self.alpha(itr, a, A, s)*delta_k,
                    apply=apply))
            second_site = np.vstack(
                self.evaluate_fitness(
                    self._fitness[itr-1]["params"] + self.alpha(itr, a, A, s)*delta_k,
                    apply=apply))

            # Determine the update velocity
            self._fitness[itr - 1]["velocities"] = self.update_velocities(first_site, 
                                                                second_site, 
                                                                self.alpha(itr, a, A, s),
                                                                self.beta(itr, b, t),
                                                                delta_k)

            # Update the SPSA particle positions
            self._fitness[itr]["params"] = self.update_positions(
                self._fitness[itr - 1]["params"], 
                self._fitness[itr - 1]["velocities"])

            # Apply the boundary conditions if any exist
            if self._projection_fn is not None:
                self._fitness[itr]["params"] = self._projection_fn(self._fitness[itr]["params"])

            # Calculate the fitness of the new positions
            self._fitness[itr]["fitness"] = self.evaluate_fitness(self._fitness[itr]["params"], 
                                                            apply=apply)

        return self._fitness[n_spsa_iterations - 1][np.argmin(self._fitness[n_spsa_iterations - 1]['fitness'])]


    def alpha(self, k, a, A, s):
        return a / (1 + A + k)**s

    def beta(self, k, b, t):
        return b / (1 + k)**t

    def delta(self, n_particles, n_params):
        return (2 * np.round(np.random.random((n_particles, n_params)))) - 1

    def update_velocities(self, first_site, second_site, alpha, beta, delta):
        return delta * beta * (first_site - second_site) / (2* alpha)
    
    def particles_dt(self):
        return np.dtype([
            ('params', np.float64, (self._n_free_params,)),
            ('velocities', np.float64, (self._n_free_params,)),
            ('fitness', np.float64)])

            
class Fitness(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def __call__(self, params):
        pass

class HeuristicPerformanceFitness(Fitness):
    def __init__(self, param_names, evaluation_function=None,
                 *args, **kwargs):
        try:
            self._heuristic_class = kwargs['heuristic_class']
            del kwargs['heuristic_class']
        except:
            raise NotImplementedError("No heuristic class was passed.")

        self._args = args
        self._kwargs = kwargs
        self._param_names = param_names
        if evaluation_function is None:
            self._evaluation_function = lambda performance: performance['loss'][:,-1].mean(axis=0)
        else:
            self._evaluation_function = evaluation_function
        
    def __call__(self, params):
        performance = perf_test_multiple(
            *self._args,
            heuristic_class = partial(
                self._heuristic_class, 
                **{name: param
                    for name, param in zip(self._param_names, params)
                }),
                **self._kwargs
        )
        
        return self._evaluation_function(performance)