#!/usr/bin/env python
# coding: utf-8
"""
    Copyright (C) 2017  Anton Crombach (anton.crombach@college-de-france.fr)
                        Alice L'Huillier (lhuillie@magbio.ens.fr)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

SYNOPSIS

    lattice_genome.py [-h,--help] [-v,--verbose] [--version] [-c,--config]
        [-s, --save]

DESCRIPTION

    I am designing and implementing lattice genomes for exploratory studies 
    into genome folding. As lattice genomes are computationally quite light, 
    they are perfect for use in education (i.e. on older computers and 
    not-so-powerful laptops).

EXAMPLES

    python lattice_genome.py -c simple_config.json
"""

# Compatible Python 2 and 3 code
from __future__ import (absolute_import, division, print_function, 
    unicode_literals)
from builtins import *

import sys
import os
import traceback
import optparse
import time
import collections
import bisect

# JSON is the file format for reading the configuration and writing results
import simplejson as json
# NumPy and SciPy do the heavy computation
import numpy as np
import numpy.random as npr
import scipy.spatial as spsp
# Matplotlib visualizes the polymer on the lattice and its statistics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Constants
#
# RGBa (red, green, blue, alpha) for the polymer line
POLYMER_BLUE = (0.125, 0.469, 0.707, 0.400)
TFACTORY_BLACK = (0.100, 0.050, 0.050, 0.700)

# Colour map, consisting of blue (n), orange (t), and purple (e)
MONOMER_COLOUR_MAP = {
    'n': (0.125, 0.469, 0.707, 0.700),
    't': (1.000, 0.498, 0.055, 0.700),
    'e': (0.593, 0.305, 0.637, 0.900),
    'r': (1.000, 0.000, 1.000, 0.900),
    'c': (0.750, 0.498, 0.055, 0.700)
    }


# Classes
#
# The elements from which we build a genome
NeutralElement = collections.namedtuple('NeutralElement', [])
TranscribedElement = collections.namedtuple('TranscribedElement', [])
EnhancerElement = collections.namedtuple('EnhancerElement', [])
CTCFElement = collections.namedtuple('CTCFElement', ['name'])

# Transitory ring elements
RingElement = collections.namedtuple('RingElement', ['name'])

# Additional enhancer elements
Enhancer_AElement = collections.namedtuple('Enhancer_AElement', [])
Enhancer_BElement = collections.namedtuple('Enhancer_BElement', [])


class LatticeGenome(object):
    """Self-avoiding polymer genome on lattice."""
    def __init__(self):
        """Set attributes to default values to make sure they exist."""
        # Types of monomer, NeutralElement, TranscribedElm, EnhancerElm.
        self.polymer = []
        # Initial positions of monomers on the lattice.
        self.initial_positions = []
        self._positions_from_file = False

        self.idx_turn = 0
        self.large_rotation_probability = 0.0

        # Storing informations about dynamic enhancers
        self.idx_enhancer_a = []
        self.idx_enhancer_b = []


    def json_decode(self, conf):
        """Build genome from json dict."""
        # Reset polymer and positions to be empty
        # AL: a quoi cela sert-il ? Les listes ne sont-elles pas déjà vides 
        #     juste après la création de l'objet ?
        # AC: yes, they should be empty already. I'm just making sure they 
        #     are, regardless of what other code we may add later.        
        self.polymer = []
        self.initial_positions = []

        if options.verbose:
            print("# Reading genome...")
        try:
            for i, elm in enumerate(conf['genome']):
                # Each element has a type, and may have a two-dimensional (xy) 
                # position.
                if elm['type'] == 'neutral':
                    self.polymer.append(NeutralElement())
                    try:
                        self.initial_positions.append(
                            [int(elm['x']), int(elm['y'])])
                    # Si le fichier de config ne contient pas de position xy, 
                    # le programme continue. Elle seront générées 
                    # aléatoirement par la suite.
                    except KeyError:
                        pass

                elif elm['type'] == 'transcribed':
                    self.polymer.append(TranscribedElement())
                    try:
                        self.initial_positions.append(
                            [int(elm['x']), int(elm['y'])])
                    except KeyError:
                        pass

                elif elm['type'] == 'enhancer':
                    self.polymer.append(EnhancerElement())
                    try:
                        self.initial_positions.append(
                            [int(elm['x']), int(elm['y'])])
                    except KeyError:
                        pass

                elif elm['type'] == 'enhancer_a':
                    self.polymer.append(Enhancer_AElement())
                    try:
                        self.initial_positions.append(
                            [int(elm['x']), int(elm['y'])])
                        self.idx_enhancer_a.append(i)
                    except KeyError:
                        pass
                    
                elif elm['type'] == 'enhancer_b':
                    self.polymer.append(Enhancer_BElement())
                    self.idx_enhancer_b.append(i)
                    try:
                        self.initial_positions.append(
                            [int(elm['x']), int(elm['y'])])
                    except KeyError:
                        pass

                elif elm['type'] == 'ctcf':
                    self.polymer.append(CTCFElement(name='ctcf'))
                    try:
                        self.initial_positions.append(
                            [int(elm['x']), int(elm['y'])])
                    except KeyError:
                        pass

                else:
                    print('Monomer {0} is missing, as its type is unknown.'.format(i))

        except KeyError:
            print('EE Incorrect genome, perhaps unknown element')

        # I assume we have read all positions if the array is of equal length 
        # as the polymer one. Then, convert positions to np.array
        if len(self.polymer) == len(self.initial_positions):
            self.initial_positions = np.array(self.initial_positions)
            self._positions_from_file = True

        # Check if a and b enhancers occur equally often.
        if len(self.idx_enhancer_a) != len(self.idx_enhancer_b) :
            print('WW Enhancers with different length can not be dynamic')
                
        try :
            self.large_rotation_probability = \
                conf['large_rotation_probability']
            if not (0.0 < self.large_rotation_probability < 1.0):
                raise ValueError(
                    'EE large scale rotation probability out of range')
        except KeyError:
            print("WW Missing large scale rotations")
            self.large_rotation_probability = 0.0


    def json_encode(self, positions):
        """Write genome to json dict."""
        # Ecrire les fichier de sortie dans le même format que les fichiers de
        # configuration, pq créer des collections.namedtuple si on l'utilise 
        # en tant que chaîne de caractères, et comment fonctionne la fonction 
        # type (renvoie type au lieu de TranscribedElement par ex).
        result = {
            'genome': [],
            'large_rotation_probability': self.large_rotation_probability
            }

        # Fill the genome
        for e, xy in zip(self.polymer, positions):
            try:
                result['genome'].append({'type': e.name.lower(), 
                    'x': xy[0], 'y': xy[1]}) 
            except AttributeError :
                result['genome'].append(
                    {'type': type(e).__name__[0:-7].lower(), 
                    'x': xy[0], 'y': xy[1]})
        return result


    def get_polymer_types_abbreviated(self):
        """
        Return polymer as a list of 'n's and 't's, respectively neutral and 
        transcribed monomers.
        """
        return [type(e).__name__[0].lower() for e in self.polymer]


    def has_positions_from_file(self):
        return self._positions_from_file


    def random_initial_positions(self):
        """
        Set initial position of each of the monomers. Default is a random 
        configuration.
        """
        def __next_direction(xy):
            """
            Auxilary function to choose a step on the lattice. The old 
            direction is given by 'xy', and the new one will be either 
            continueing in the same direction, or turning left/right.
            """
            p, q = tuple(xy)
            nbh = [[-q, p], [p, q], [q, -p]]
            return nbh[npr.randint(3)]

        # First monomer is positioned at [0,0], second [x,y] takes one step 
        # north, south, east on the lattice.
        x = npr.randint(2)
        y = npr.choice([-1, 1]) if x == 0 else 0
        
        # Two positions done, let's do the rest
        directions = [[0, 0], [x, y]]
        for _ in self.polymer[2:]:
            directions.append(__next_direction(directions[-1]))

        # Input to cumsum is a list, returns a numpy array
        self.initial_positions = np.cumsum(directions, axis=0)


    def long_range_interactions(self):
        """
        Return for each monomer, all monomers that it forms a long range 
        interaction with.
        """
        # return self._kdtree.query_ball_point(self.positions, 3, p=1.0)
        pass


    def attempt(self, positions):
        """
        Make a "move" on a copy of the polymer. Three possible moves are 
        defined: the end-points can wiggle 90 degrees, a turn can flip-flop, 
        and polymer arms can rotate 90 degrees. Of course, not doing any move
        is also possible.

        The first two moves are known as Mover set 1 (MS1) of Chan & Dill 
        1993, 1994. The "large" rotation is part of Mover set 2 (MS2).
        """
        aux_positions = np.copy(positions)
        self.idx_turn = -1
        # Pick a monomer
        idx = npr.randint(len(aux_positions))

        # End points wiggle
        if idx == 0:
            aux_positions[0] = self.__neighbour(aux_positions[1])
        elif idx == len(aux_positions) - 1:
            aux_positions[-1] = self.__neighbour(aux_positions[-2])
        else:
            # `idx' refers to a monomer that is not an end-point.

            # Is the monomer in a 90 degree turn?
            turn = (aux_positions[idx-1] + aux_positions[idx+1]) \
                    - 2 * aux_positions[idx]
            # If x and y are unequal to zero, flip the turn
            if np.all(turn != 0):
                aux_positions[idx] += turn
                self.idx_turn = idx
            else:
                # Perhaps do a "large scale" rotation
                rr = npr.random()
                if rr < self.large_rotation_probability:
                    # Which of the two genome arms?
                    half = self.large_rotation_probability / 2.0
                    iv = (0, idx) if rr < half else (idx+1, len(aux_positions))
                    # Rotate left or right?
                    quarter = half / 2.0
                    lr = "left" if rr < quarter or \
                        half+quarter <= rr < self.large_rotation_probability else "right"
                    # Do the rotation
                    aux_positions = self.__pivot(aux_positions, idx, iv, lr)
                else:
                    # Do nothing if r > self.large_rotation_probability
                    pass

        return aux_positions


    def has_transcription_loop_transcription(self, gpos, fpos):
        """Check if genome has two regions transcribed with a loop between."""
        # Factory positions as set for quick look-up
        fpos = set((x,y) for x, y in fpos.tolist())

        # Walk along polymer, find a t-l-t structure
        last_t_found, last_l_found = -1, -1
        for i, elm in enumerate(self.polymer):
            # We're looking for transcribed regions on top of tfactory elements
            flag_transcribed = (type(elm).__name__[:-7] == 'Transcribed' and 
                tuple(gpos[i]) in fpos)
            # and some non-transcribed elements that are not on the tfactory, 
            # i.e. "loops"
            flag_loop = (type(elm).__name__[:-7] != 'Transcribed' and 
                tuple(gpos[i]) not in fpos)

            if flag_transcribed and last_l_found == -1:
                # We found a transcribed region overlapping with tfactory, but
                # we have not yet found a "loop" part
                # print("Found first transcribed overlapping with tfactory")
                last_t_found = i

            elif flag_transcribed and last_l_found != -1:
                # Yay! We found another transcribed region on top of tfactory,
                # and we already had a "loop" part
                print("Found transcribed-loop-transcribed at", last_t_found,
                    last_l_found, i)
                return True

            elif flag_loop and last_t_found != -1:
                # We found a neutral (or other element) that is not overlapping
                # with the tfactory
                # print("Found neutral, not overlapping with tfactory")
                last_l_found = i

        return False


    def __neighbour(self, xy):
        """Return one of four possible neighbours."""
        offset = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        return xy + offset[npr.randint(len(offset))]


    def __pivot(self, aux_positions, idx, interval, lr):
        """Rotate part of the polymer 90 degrees left or right."""
        pivot = aux_positions[idx]
        lower, upper = interval
        
        # Translate, so pivot point is at (0, 0).
        aux = np.array([p - pivot for p in aux_positions[lower:upper]])
        # Rotate. Depending on the left/right rotation, the x or y coordinate 
        # needs to be negated.
        if lr == 'left':
            aux[:, 0], aux[:, 1] = aux[:, 1], -aux[:, 0].copy()
        else:
            aux[:, 0], aux[:, 1] = -aux[:, 1], -aux[:, 0].copy()
        # Translate back to original coordinates
        aux += pivot
        # Replace the old polymer segment with the rotated one
        aux_positions[lower:upper] = aux
        return aux_positions


    def element_type(self, idx):
        return type(self.polymer[idx]).__name__[:-7].lower()


    def __len__(self):
        return len(self.polymer)


    def __str__(self):
        return "-".join(self.get_polymer_types_abbreviated())


class TranscriptionFactory(object):
    """
    A transcription factory is a membrane-less organelle in the nucleus
    that concentrates RNA polymerase, splicing machinery, transcription factors
    and other components.

    Here it is simply a big circle (default radius 3) that has affinity for 
    transcribed regions of the genome.
    """
    def __init__(self):
        # Positions of monomers on the lattice.
        self.center = (0, 0)
        self.radius = 0
        self.initial_positions = []
        self._positions_from_file = False

    def json_decode(self, conf):
        """Read in configuration."""
        try:
            tfac = conf['transcription_factory']
            self.center = (tfac['x'], tfac['y'])
            self.radius = tfac['radius']
        except KeyError:
            print("WW Missing transcription factory.")

    def json_encode(self):
        """Return a dictionary that is automatically converted into json."""
        return { 'x': self.center[0],
                 'y': self.center[1],
                 'radius': self.radius}

    def random_initial_positions(self):
        """Generate a discretized circle."""
        x, y = np.mgrid[-self.radius:self.radius+1, -self.radius:self.radius+1]
        circle = x**2 + y**2 < self.radius**2
        self.initial_positions = np.stack((x[circle], y[circle]), axis=1)

    def element_type(self, idx):
        return "transcription_factory"

    def __len__(self):
        return len(self.initial_positions)

    def __str__(self):
        return "tfactory"


class World(object):
    """
    A simplified nuclear environment. Currently, the nucleus is a lattice with 
    two particles, namely the genome and a transcription factory. In general, 
    particles are composed of one or more sites on a grid. Energy levels and 
    what "move" happens next, is computed from the grid sites associated to an 
    object.

    There are two approaches to implement the grid and its particles:

    1. Explicitly store the lattice as a 2D integer array, where each site has 
       an integer (unique?) which in turn refers to a monomer, a component of 
       the transcription factory, or another component. Next, a dictionary maps
       integers to particles.

    (+) After an initial scan of the entire grid and calculating the starting 
        energy of the system, all updates are local --- which is fast.
    (+) The two monomer moves are local and easy to perform. 
    (-) Adding more complex (non-local) moves is difficult (e.g. rotations of
        parts of the polymer). Perhaps they are easier if we let go of the rule
        that all monomers are in the von Neuman neighbourhood (n, s, e, w).
    (-) If the ratio occupied/empty is low, many random site updates will not 
        do anything (i.e. wasting random numbers).

    2. Store object configuration on the lattice as a list of sites. We need 
       to keep track of which list intervals refer to which object -- 
       similarly to the dict mapping of (1).

    (+) Easily converted to a more continuous approach.
    (+) Different "moves", even exotic ones, are relatively easy.
    (-) KDTree (rebuilding) needed to speed up finding of neighbours locally.
    (-) Calculating energy is a global affair.

    The main issue with using a lattice is that single monomer bonds are 
    immutable -- unless I loosen that constraint --- which hampers the 
    polymers flexibility. Are there alternative schemes that allow a more 
    flexible polymer embedded on a lattice?
    """
    def __init__(self):
        self.end_time = 100
        self.stats_time = 10

        self.genome = None
        self.transcription_factory = None

        # List of positions on the lattice taken by genome, transcription 
        # factory, or some other particle.
        self.positions = []
        self.particles = None
        self.weights = None
        # Energy accumulated in the entire system
        self.energy = 0.0
        # Temperature of world
        self.temp = 0.0
        # Mapping interactions between particles to energy
        self.interactions = {}
        # Storing informations about cohesin ring
        self.cohesin_ring_formation_probability = 0
        self.ring_element_idx = []
        self.loops_nbr = 0
        self.max_number_of_loops = 50
        # Storing informations about dynamic enhancers
        self.shifting_enhancers_flag = 0
        self.positions_enhancer_a = []
        self.positions_enhancer_b = []

        # Alternative stopping criterion
        self.stop_if_loop_formed = False

        # Binary tree that keeps track of monomers and other elements in 
        # space. Used to quickly look up neighbours. About 4--5 times faster 
        # than naive all-against-all calculation.
        self._kdtree = None


    def json_decode(self, conf):
        """Read in simulation parameters."""
        try:
            self.end_time = conf['end_time']
            self.stats_time = conf['observe_time']
        except KeyError:
            print("EE Missing time.")

        try:
            self.temp = conf['temperature']
        except KeyError:
            print('EE Missing temperature.')

        # Read in interactions, if combinations are missing we take them to
        # be like neutral vs. neutral (if that one is missing, an error is
        # flagged).
        try:
            for iaction in conf['interactions']:
                first, second = iaction['first'], iaction['second']
                if first > second:
                    first, second = second, first
                self.interactions[(first, second)] = \
                    np.array(iaction['distance_to_energy'])
        except KeyError:
            print("WW Interactions not defined properly.")

        # Confine polymer to a radius around the transcription factory.
        try:
            self.confinement = conf['confinement']['radius']
            self.penalty = conf['confinement']['penalty']
        except KeyError:
            print("WW confinement not defined properly.")

        # Cohesin rings may be used -- and thus should be read
        try:
            self.cohesin_ring_formation_probability = \
                conf['cohesin_ring_formation_probability']
            if not(0.0 <= conf['cohesin_ring_formation_probability'] <= 1.0):
                raise ValueError('Probability of cohesin ring formation is out of range') 

        except KeyError :
            print("WW Missing ring complexes")

        # AC: I've removed the code for implicit changes of the probability. I
        # prefer that the program stops and the user fixes the config file.
        
        try:
            self.max_number_of_loops = conf['max_number_of_loops']
        except KeyError:
            print("WW number of loops is limited to 50")
            
        # AC: If we only allow for 0 or 1, we better call it a "flag" that 
        # toggles on/off.
        try:
            self.shifting_enhancers_flag = bool(
                conf['shifting_enhancers_flag'])
        except KeyError:
            print('WW Missing shifting enhancers flag')

        # Alternative stopping
        try:
            self.stop_if_loop_formed = bool(conf['stop_if_loop_formed'])
            print('WW Simulation stops as first loop is formed')
        except KeyError:
            print('WW Simulation continues until end_time.')


    def json_encode(self):
        """Write simulation parameters to dict."""
        return {
            'end_time': self.end_time,
            'observe_time': self.stats_time,
            'temperature': self.temp,
            'cohesin_ring_formation_probability':
                self.cohesin_ring_formation_probability,
            'shifting_enhancers_flag': self.shifting_enhancers_flag,
            'genome': 
                self.genome.json_encode(self.genome_positions())['genome'],
            'transcription_factory': self.transcription_factory.json_encode(),
            'interactions': [{
                "first": k[0], 
                "second": k[1], 
                "distance_to_energy": v.tolist()}
                    for k, v in self.interactions.iteritems()],
            'confinement': {
                'radius': self.confinement,
                'penalty': self.penalty}
            }


    def add_genome(self, genome):
        """Add genome particle, a polymer."""
        self.genome = genome
        # If it does not have an initial position yet, generate one.
        if not self.genome.has_positions_from_file():
            self.genome.random_initial_positions()


    def add_transcription_factory(self, factory):
        """Add transcription factory particle, a big circle."""
        self.transcription_factory = factory
        # Factory by definition does not have an initial position yet.
        # Let's generate one
        self.transcription_factory.random_initial_positions()


    def prepare(self):
        """Prepare simulation."""
        self.particles = [self.genome, self.transcription_factory]
        self.weights = np.cumsum([len(p) for p in self.particles])
        self.positions = np.concatenate((self.genome.initial_positions, 
            self.transcription_factory.initial_positions))
        self.energy, self._kdtree = self.__calculate_energy(
            self.positions)
        self.positions_enhancer_a, self.positions_enhancer_b = \
            self.new_positions_enhancers()        


    def simulate(self, observers):
        """Monte Carlo simulation algorithm."""
        time = 0
        self.prepare()

        # And go
        while time < self.end_time:
            # Output statistics
            if time % self.stats_time == 0:
                observers.observe(time, self)
                
            if not observers.pause:
                # A single simulation step is defined as to attempt to "move"
                # each element of the simulation. (At the moment only the 
                # genome.)
                self.step(time)
                time += 1

            # Alternative stopping criterion
            if self.stop_if_loop_formed:
                if self.genome_has_loop():
                    self.end_time = time

        # Final observation
        observers.observe(time, self)


    def step(self, time_step):
        """
        Accept attempt if it lowers the energy or simply by chance using 
        an exponential distribution.
        """
        new_positions, new_energy, new_tree = self.attempt()
        delta_e = new_energy - self.energy
        if delta_e < 0.0 or npr.random() <= np.exp(-(delta_e / self.temp)):
            # Biased acceptance of attempts
            self.positions = new_positions
            self.energy = new_energy
            self._kdtree = new_tree

            # If we accept, we always try cohesin ring formation.
            if self.cohesin_ring_formation_probability > 0.0:
                self.cohesin_ring_sliding()
                # If we just had a turn...
                if self.genome.idx_turn != -1:
                    self.cohesin_ring_formation(self.genome.idx_turn)

            # And we always try to shift enhancers
            if self.shifting_enhancers_flag:
                # Update positions ..
                self.positions_enhancer_a, self.positions_enhancer_b = \
                    self.new_positions_enhancers()
                # .. and shift the enhancers
                if sorted(self.positions_enhancer_a) == sorted(self.positions_enhancer_b):
                    # print(self.positions_enhancer_a, self.positions_enhancer_b)
                    # print(self.genome.idx_enhancer_a, self.genome.idx_enhancer_b)
                    self.new_profile_enhancers()
                    # print(self.positions_enhancer_a, self.positions_enhancer_b)

            self.energy, self._kdtree = self.__calculate_energy(self.positions)


    def genome_has_loop(self):
        """Establish if the genome has formed a loop."""
        if self.shifting_enhancers_flag:
            if sorted(self.positions_enhancer_a) == sorted(self.positions_enhancer_b):
                # Enhancers align
                return True

        elif self.loops_nbr > 0:
            # If we have cohesin rings, we have loops
            return True

        else:
            # If we have a transcription factory, two separate sites of the
            # genome should attach to it to form a loop.
            return self.genome.has_transcription_loop_transcription(
                self.positions[0:self.weights[0]],
                self.transcription_factory.initial_positions)


    def cohesin_ring_formation(self, idx):
        """Attempt to form a cohesin ring at position idx."""
        if self.loops_nbr < self.max_number_of_loops:
            rr = npr.random()
            if rr < self.cohesin_ring_formation_probability:
                try:
                    # If we have a "hairpin", we can form a loop
                    if list(self.positions[idx]) == list(self.positions[idx-2]):

                        # Update loop number and make a name out of it
                        self.loops_nbr += 1
                        loopie = 'ring{0}'.format(self.loop_nbr)

                        self.genome.polymer[idx] = RingElement(name=loopie)
                        self.genome.polymer[idx+1] = RingElement(name=loopie)
                        self.genome.polymer[idx-2] = RingElement(name=loopie)
                        self.genome.polymer[idx-3] = RingElement(name=loopie)
                        self.ring_element_idx.append([idx-3, idx-2, idx, idx+1])

                        self.interactions[(loopie, loopie)] = \
                            self.interactions[('ring','ring')]

                        # print(self.ring_element_idx)
                        # print(self.interactions)
                        # print(self.loops_nbr)

                    elif list(self.positions[idx]) == list(self.positions[idx+2]):

                        # Update loop number and make a name out of it
                        self.loops_nbr += 1
                        loopie = 'ring{0}'.format(self.loop_nbr)

                        self.genome.polymer[idx-1] = RingElement(name=loopie)
                        self.genome.polymer[idx] = RingElement(name=loopie)
                        self.genome.polymer[idx+2] = RingElement(name=loopie)
                        self.genome.polymer[idx+3] = RingElement(name=loopie)                       
                        self.ring_element_idx.append([idx-1, idx, idx+2, idx+3])
                        
                        self.interactions[(loopie, loopie)] = \
                            self.interactions[('ring','ring')]

                        # print(self.ring_element_idx)
                        # print(self.interactions)
                        # print(self.loops_nbr)

                except IndexError:
                    pass

    
    def cohesin_ring_sliding(self) :
        """Advance any of the cohesin rings, until they bump into CTCF."""
        # Little helper functions to make the code more elegant
        def is_ctcf(x):
            return type(self.genome.polymer[x]).__name__ == 'CTCFElement'

        def is_ring(x):
            return type(self.genome.polymer[x]).__name__ == 'RingElement'

        # And now the real code of this function...
        for i, relm in enumerate(self.ring_element_idx):
            try:
                a, b, c, d = relm
                if ([list(self.positions[a]), list(self.positions[b])] == 
                    [list(self.positions[d]), list(self.positions[c])]):

                    # print([list(self.positions[a]), list(self.positions[b])],
                    #     [list(self.positions[c]), list(self.positions[d])])
                    if is_ctcf(a):
                        if is_ctcf(d):
                            self.genome.polymer[b] = NeutralElement()
                            self.genome.polymer[c] = NeutralElement()
                            # del relm[1], relm[2]

                    elif is_ring(a):
                        try:
                            if a-1 < 0:
                                raise IndexError('Cohesin ring can not slide from one DNA end to the other')

                            if is_ctcf(a-1):
                                self.genome.polymer[b] = NeutralElement()
                                self.genome.polymer[a-1] = \
                                    CTCFElement(name='ctcf_ring')
                                # relm[0], relm[1] = a-1, a

                            else:
                                self.genome.polymer[a-1] = \
                                    RingElement(name='ring' + str(i+1))
                                self.genome.polymer[b] = NeutralElement()
                                # relm[0], relm[1] = a-1, a

                        except IndexError:
                            pass
                    
                    if is_ctcf(d):
                        # We don't care if d is a CTCF element.
                        pass

                    elif is_ring(d):
                        try:
                            if d + 1 > len(self.genome) - 1:
                                raise IndexError('Cohesin ring can not slide from one DNA end to the other')

                            if is_ctcf(d+1):
                                self.genome.polymer[c] = NeutralElement()
                                self.genome.polymer[d+1] = \
                                    CTCFElement(name='ctcf_ring')
                                # relm[2], relm[3] = d, d+1

                            else:
                                self.genome.polymer[d+1] = \
                                    RingElement(name='ring' + str(i+1)) 
                                self.genome.polymer[c] = NeutralElement()
                                # relm[2], relm[3] = d, d+1

                        except IndexError:
                            pass

                    # print(self.ring_element_idx)

            except ValueError:
                pass
  

    def new_profile_enhancers(self):
        """
        We know that the two enhancers (a and b) overlap. Such a configuration 
        can refer to 2 types of loops. Here we are only interested in loops in 
        which the last residu of an enhancer interacts with the first of the 
        other.

        If enhancers are not located at the end of the DNA strand, enhancers 
        are shifted in this direction.
        """
        try:
            # No matter the relative position of enhancers, we can check if 
            # the loop type is the interesting one.
            if (list(self.positions[max(self.genome.idx_enhancer_a)]) == 
                list(self.positions[min(self.genome.idx_enhancer_b)])):
                
                try:
                    # Take the furthest monomer, starting from index 0
                    #
                    # AC: I think there is a bug here, since we always subtract
                    # the length of idx_enhancer_a (also if we took the index 
                    # from the b list).
                    furthest = max(
                        self.genome.idx_enhancer_a + self.genome.idx_enhancer_b)
                    ii = furthest + 1
                    jj = furthest - len(self.genome.idx_enhancer_a) + 1

                    # Swap monomers (i.e. only types are swapped)
                    self.genome.polymer[ii], self.genome.polymer[jj] = \
                        self.genome.polymer[jj], self.genome.polymer[ii]
                
                    # Take the closest monomer, starting from index 0
                    #
                    # AC: Same bug...
                    closest = min(
                        self.genome.idx_enhancer_a + self.genome.idx_enhancer_b)
                    pp = closest - 1
                    qq = closest + len(self.genome.idx_enhancer_a) - 1

                    # Swap other side of the loop as well
                    self.genome.polymer[pp], self.genome.polymer[qq] = \
                        self.genome.polymer[qq], self.genome.polymer[pp]                        
                    
                    # Update the cache of indices
                    if (max(self.genome.idx_enhancer_a, 
                        self.genome.idx_enhancer_b) == 
                            self.genome.idx_enhancer_a):

                        for i in range(len(self.genome.idx_enhancer_a)):
                            self.genome.idx_enhancer_a[i] += 1
                        for j in range(len(self.genome.idx_enhancer_b)):
                            self.genome.idx_enhancer_b[j] -= 1

                    else:

                        for i in range(len(self.genome.idx_enhancer_a)):
                            self.genome.idx_enhancer_a[i] -= 1
                        for j in range(len(self.genome.idx_enhancer_b)):
                            self.genome.idx_enhancer_b[j] += 1

                    # For debugging
                    # print(self.genome.idx_enhancer_a, self.genome.idx_enhancer_b)
                except IndexError:
                    pass
                
        except ValueError:
            pass


    def new_positions_enhancers(self):
        """Update the positions of a and b enhancers."""
        new_positions_enhancer_a = []
        new_positions_enhancer_b = []
        try :
            # Use the genome's indices of enhancers to update their positions.
            for i in self.genome.idx_enhancer_a :
                new_positions_enhancer_a.append(list(self.positions[i]))

            for j in self.genome.idx_enhancer_b :
                new_positions_enhancer_b.append(list(self.positions[j]))

        except IndexError:
            # AC: Not sure when this tends to fail...
            pass

        return new_positions_enhancer_a, new_positions_enhancer_b


    def attempt(self):
        """
        Choose a particle and let it do an attempt. Currently, choosing a 
        particle is easy, as only the genome moves.
        """
        new_genome_positions = self.genome.attempt(
            self.positions[0:self.weights[0]])

        # Get all positions together to calculate energy
        aux_positions = np.concatenate((new_genome_positions, 
            self.transcription_factory.initial_positions))

        energy, tree = self.__calculate_energy(aux_positions)
        return aux_positions, energy, tree


    def genome_positions(self):
        """Return only positions of the polymer genome."""
        return self.positions[0:self.weights[0]]


    def transcription_factory_positions(self):
        """Return positions of the tfactory."""
        return self.positions[self.weights[0]:]


    def __calculate_energy(self, aux_positions):
        """
        Calculate energy. 

        A self-avoiding polymer defines an energy 'penalty' between 
        non-adjacent monomers that touch each other. Here I define three 
        energy levels:

        - 10: monomers are on top of each other.
        - 1 : monomers are next to each other.
        - 0.1: monomers are diagonally close.

        These energy levels should lead to polymers that become elongated.

        NEW: distance to energy is now defined in the configuration file, and
        accessed through the `interactions' attribute.
        """
        # Start with zero energy.
        energy = 0
        
        # KDTree expects all positions of the monomers, a max distance (2), and
        # a distance measure (1.0 = Manhattan distance).
        kdtree = spsp.cKDTree(aux_positions, compact_nodes=False, 
            balanced_tree=False)
        neighbours = kdtree.query_ball_point(aux_positions, 
            len(any_elem(self.interactions))-1, p=1.0)
        
        # Iterate all elements of particles and their lists of neighbours
        for i, pnbs  in enumerate(zip(aux_positions, neighbours)):
            # p is the current element, nbs is a list of its neighbours (these
            # are indices to elements in aux_positions)
            p, nbs = pnbs
            # Neighbour distance {0, 1, .., 4} to energy = {10, 1, 0.1} etc. 
            # j is the index to a neighbour, whose position we retreive 
            for j in nbs:
                if i != j:
                    # Which particle do the indices belong to?
                    pi = bisect.bisect(self.weights, i)
                    pj = bisect.bisect(self.weights, j)

                    # Which element (type of monomer) do the indices refer to?
                    offset_i = self.weights[pi-1] if pi > 0 else 0
                    offset_j = self.weights[pj-1] if pj > 0 else 0
                    try:
                        # CTCF and Ring elements have a name
                        first = self.particles[pi].polymer[i-offset_i].name
                        second = self.particles[pj].polymer[j-offset_j].name

                    except AttributeError:
                        # Default behaviour
                        first = self.particles[pi].element_type(i - offset_i)
                        second = self.particles[pj].element_type(j - offset_j)

                    # Maintain alphabetical ordering of interaction tuple
                    if first > second:
                        first, second = second, first

                    # Calculate energy of interaction
                    try:
                        energy += self.interactions[(first, second)][
                            np.sum(np.abs(p - aux_positions[j]))]
                    except KeyError:
                        energy += self.interactions[("neutral", "neutral")][
                            np.sum(np.abs(p - aux_positions[j]))]
        
        # A mild form of confinement to make our polymer not drift away
        energy += self.penalty * np.count_nonzero(
            [np.dot(p, p) > self.confinement**2 for p in aux_positions])

        # Done.
        return energy, kdtree


class Observers(object):
    """Observer manager."""
    def __init__(self):
        # Running or pausing
        self.pause = False
        # Graphics
        self._first = True
        self._fig = None
        self._axs = None
        
        # Polymer line segments and monomer positions
        self._polymer_line = None
        self._monomers = None
        self._tfactory = None
        # Energy over time line, with a label tracking the latest energy
        self._energy_line = None
        self._energy_label = None

        # Contact matrix
        self._contacts = None

        # Tracking polymer positions over time, save to file
        self.polymer_positions = {}
        self.polymer_positions_fname = ''

        # Take a screenshot
        self.screenshot_fname = ''


    def json_decode(self, conf):
        """Reading in which data to collect and save to file."""
        # At the moment we only track polymer positions
        try:
            self.polymer_positions_fname = \
                conf['observers']['polymer_positions']
            self.screenshot_fname = conf['observers']['screenshot']
        except KeyError:
            # Fail silently, because we do not *have* to track polymer 
            # positions.
            pass


    def json_encode(self):
        """Write out polymer positions per time step."""
        return [{'time': t, 'positions': p.tolist()} 
            for t,p in sorted(self.polymer_positions.iteritems())]


    def observe(self, time_step, world):
        """
        Connect observers to the right parts of the world and let them compute 
        their statistics.
        """
        # Get polymer data, first energy then positions
        en = world.energy
        xy = world.genome_positions()
        # nbs = world.genome.long_range_interactions()
        # Get tfactory data
        tf = world.transcription_factory_positions()

        # Store positions for writing to file later
        if self.have_results_to_save:
            self.polymer_positions[time_step] = np.copy(xy)

        # Reshape into sequence of line segments [[(x0,y0),(x1,y1)],...]
        xy = xy.reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])
        tf = tf.reshape(-1, 1, 2)

        if self._first:
            # Set up visualization only once
            self._first = False

            # Create grid of 2 rows, 3 columns
            self._fig = plt.figure(figsize=(9,6))
            gs = gridspec.GridSpec(2, 3)
            gs.update(left=0.02, right=0.98, wspace=0.15, hspace=0.15)
            self._axs = [
                self._fig.add_subplot(gs[:, :2]),
                self._fig.add_subplot(gs[0, 2]),
                self._fig.add_subplot(gs[1, 2])]

            # Axis 0: 2D polymer conformation
            self.__prepare_polymer_plot(time_step, world, xy, segments, tf)

            # Axis 1: energy of the polymer over time
            self.__prepare_energy_plot(time_step, en)

            # Axis 2: distance matrix
            self.__prepare_pairwise_distance_plot(time_step, 
                world.genome_positions())
            
            # Connect key press and mouse button events to pause simulation.
            self._fig.canvas.mpl_connect("key_press_event", 
                self.__pause_simulation)
            self._fig.canvas.mpl_connect("button_press_event", 
                self.__pause_simulation)

            # Preparations done
            self._fig.show()

        else:
            # Update time and polymer positions
            self.__observe_polymer(time_step, world, xy, segments)
            # Update time vs energy graph
            self.__observe_energy(time_step, en)
            # Update distance matrix
            self.__observe_pairwise_distances(time_step, 
                world.genome_positions())
            # Update time and long range interactions
            # self.__observe_long_range_interactions(time_step, nbs)

            self._fig.canvas.draw()
            self._fig.canvas.flush_events()


    def take_screenshot(self):
        """Save a high-res PNG of the final figure displayed on the screen."""
        self._fig.savefig(self.screenshot_fname, dpi=300)


    def have_results_to_save(self):
        """Quick check if we have anything to write to disk."""
        return self.polymer_positions_fname


    def __pause_simulation(self, event):
        """
        Pressing any key pauses the simulation. A second key press will start
        the simulation again.
        """
        self.pause = not self.pause
        if self.pause:
            print("# Paused...")
        else:
            print("# Running...")


    def __prepare_polymer_plot(self, time_step, world, xy, segments, tf):
        """Prepare polymer line and monomers."""
        # The polymer with its monomers is built from lines and polygons
        self._polymer_line = mpl.collections.LineCollection(segments)
        self._polymer_line.set_color(POLYMER_BLUE)

        self._monomers = mpl.collections.RegularPolyCollection(8, 
            sizes=[8.0 for _ in xy], offsets=xy, 
            transOffset=self._axs[0].transData)
        trans = mpl.transforms.Affine2D().scale(self._fig.dpi/72.0)
        self._monomers.set_transform(trans)

        self._tfactory = mpl.collections.RegularPolyCollection(5, 
            sizes=[10.0 for _ in tf], offsets=tf,
            transOffset=self._axs[0].transData)

        # Different monomers have different colours
        pt = world.genome.get_polymer_types_abbreviated()
        pc = [MONOMER_COLOUR_MAP[m] for m in pt]
        self._monomers.set_facecolor(pc)
        self._monomers.set_edgecolor(pc)

        # Transcription factory is black
        self._tfactory.set_facecolor(TFACTORY_BLACK)
        self._tfactory.set_edgecolor(TFACTORY_BLACK)

        self._axs[0].add_collection(self._polymer_line)
        self._axs[0].add_collection(self._monomers)
        self._axs[0].add_collection(self._tfactory)

        # Make the plot pretty, no annoying tick or their labels
        lim = (-25, 25)
        self._axs[0].set_xlim(*lim)
        self._axs[0].set_ylim(*lim)
        self._axs[0].set_xticks([])
        self._axs[0].set_xticklabels([])
        self._axs[0].set_yticks([])
        self._axs[0].set_yticklabels([])
        # Make sure the plot is square
        self._axs[0].set_aspect(1.0)


    def __observe_polymer(self, time_step, world, xy, segments):
        """Update time and polymer positions."""
        self._axs[0].set_title('Time = {0}'.format(time_step))
        self._polymer_line.set_paths(segments)

        # Different monomers have different colours
        pt = world.genome.get_polymer_types_abbreviated()
        pc = [MONOMER_COLOUR_MAP[m] for m in pt]
        self._monomers.set_facecolor(pc)
        self._monomers.set_edgecolor(pc)
        # And update positions
        self._monomers.set_offsets(xy)        


    def __prepare_energy_plot(self, time_step, energy):
        """Prepare to plot energy over time."""
        # Create plotting area with 1 data point (at the moment)
        self._energy_line, = self._axs[2].plot([time_step], [energy])

        # No extra spines
        self._axs[2].spines['top'].set_visible(False)
        self._axs[2].spines['right'].set_visible(False)

        # Add labels for clarity
        self._axs[2].set_xlabel('Time (au)')
        self._axs[2].set_ylabel('Energy (au)')

        # Keep track of current energy of the polymer
        self._energy_label = self._axs[2].text(.98, .98, 
            'E = {0}'.format(energy), horizontalalignment='right', 
            verticalalignment='top', transform=self._axs[2].transAxes)


    def __observe_energy(self, time_step, energy):
        """Update time vs energy graph."""
        # Label to keep track of the exact value
        self._energy_label.set_text('E = {0}'.format(energy))
        
        # It's not elegant to add a value to the plot data...
        aux_t, aux_e = self._energy_line.get_data()
        self._energy_line.set_data(
            (np.append(aux_t, time_step), np.append(aux_e, energy)))
        # Rescale etc. needed!
        self._axs[2].relim()
        self._axs[2].autoscale_view()


    def __prepare_pairwise_distance_plot(self, time_step, positions):
        """Prepare to plot distance matrices."""
        # Calculating distances, unfortunately computationally costly.
        d = spsp.distance.squareform(
                spsp.distance.pdist(positions, 'cityblock'))
        self._dist_img = self._axs[1].imshow(d, interpolation='nearest')

        # Add labels for clarity
        self._axs[1].set_title('Pairwise distance')
        self._axs[1].set_ylabel('Monomer index')

        # Add colorbar, make sure to specify tick locations
        self._fig.colorbar(self._dist_img, ticks=[0, 5, 10, 15], 
            ax=self._axs[1])


    def __observe_pairwise_distances(self, time_step, positions):
        """Update matrix of pairwise distances."""
        # Calculating distances, unfortunately computationally costly.
        d = spsp.distance.squareform(
                spsp.distance.pdist(positions, 'cityblock'))
        self._dist_img.set_data(d)


# Functions
def any_elem(d):
    """Return an element from a dictionary or list"""
    try:
        return d.itervalues().next()
    except AttributeError:
        return d[0]

def read_json_configuration(fname):
    """Read in simulation parameters (in JSON format)."""
    try:
        infile = open(fname, 'r')
    except IOError:
        print('EE Cannot find:', fname)
    else:
        with infile:
            conf = json.load(infile)
    return conf


def write_json_configuration(fname, out_data, compress=False):
    """Write out simulation parameters and results (in JSON format)."""
    try:
        outfile = open(fname, 'w')
    except IOError:
        print('EE Cannot open for writing:', fname)
    else:
        # Format json. Compress squeezes whitespace out.
        indent = None if compress else 2
        separators = (',', ':') if compress else (', ', ': ')
        # Dump a json string and write it to file.
        conf = outfile.write(
            json.dumps(out_data, indent=indent, separators=separators, 
                encoding='utf-8').decode('utf8'))


def write_simulation_results(opt, config, world, observers):
    """
    Write out configuration and simulation results. The output file can be 
    used to continue a simulation at a later point.
    """
    out_data = {'random_seed': config['random_seed']}
    out_data.update(world.json_encode())

    # Preparing file name
    outfilename = os.path.expanduser(opt.save)
    print("# Writing results to", outfilename)
    write_json_configuration(outfilename, out_data)

    # Perhaps we also need to write data picked up by observers
    if observers.have_results_to_save():
        # We only have polymer positions at the moment...
        print("# Writing data of observers")
        outfilename = os.path.expanduser(observers.polymer_positions_fname)
        out_data = observers.json_encode()
        write_json_configuration(outfilename, out_data, True)
        

def main():
    """Starting point. In general, simulations are built as follows:

    - Read in simulation parameters, including how to set up the simulation 
      run, what statistics to acquire and save;
    - Set up simulation engine (the "core" part of the code that does the 
      heavy computation)
    - Run the simulation, with collection and visualization of statistics;
    - As the simulation finishes, clean up, close files etc.
    """

    global options, args
    # Read in simulation parameters
    conf = read_json_configuration(options.config)

    # If random seed == -1, then use current time multiplied by the process 
    # number as a starting point.
    if conf['random_seed'] == -1:
        conf['random_seed'] = int(time.time() * os.getpid()) % 2**32
    npr.seed(conf['random_seed'])

    # Build a genome
    g = LatticeGenome()
    g.json_decode(conf)
    
    # Some output to see what is going on
    out = "# Simulating a polymer genome of length {0}".format(len(g))
    print(out)

    # Build a transcription factory
    f = TranscriptionFactory()
    f.json_decode(conf)

    # Build the molecular world in which the genome is placed
    w = World()
    w.json_decode(conf)
    w.add_genome(g)
    w.add_transcription_factory(f)

    # Set up which statistics to compute during simulation
    o = Observers()
    o.json_decode(conf)
    
    # Run the simulation
    w.simulate(o)
    # If you do not want the plot window to close immediately, uncomment 
    # the two lines below.
    print("# Giving you some time to enjoy the plots...")
    # Wait for given number of seconds
    time.sleep(5)

    # Write out simulation results and take a final screenshot
    write_simulation_results(options, conf, w, o)
    o.take_screenshot()
    # End of the simulation


if __name__ == '__main__':
    try:
        start_time = time.time()
        
        parser = optparse.OptionParser( 
            formatter=optparse.TitledHelpFormatter(), 
            usage=globals()['__doc__'], version='Unknown')
        parser.add_option('-v', '--verbose', action='store_true',
            default=False, 
            help='verbose output')
        parser.add_option('-c', '--config', type='string', 
            default='conf.json', 
            help='configuration file name')
        parser.add_option('-s', '--save', type='string', 
            default='saved.json', 
            help='save final state of simulation to file name')
        (options, args) = parser.parse_args()
        
        #if len(args) < 1:
        #    parser.error('missing argument')

        if options.verbose: print(time.asctime())
        main()
        if options.verbose: print(time.asctime())
        if options.verbose: print('# total time (min):', end='')
        if options.verbose: print((time.time() - start_time) / 60.0)
        sys.exit( 0 )

    except KeyboardInterrupt as e: # Ctrl-C
        raise e

    except SystemExit as e: # sys.exit()
        raise e

    except Exception as e:
        print('Error, unexpected exception')
        print(str(e))
        traceback.print_exc()
        os._exit(1)
