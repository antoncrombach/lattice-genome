#!/usr/bin/env python
# coding: utf-8
"""
    Copyright (C) 2017  Anton Crombach (anton.crombach@college-de-france.fr)

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


# Constants
#
# RGBa (red, green, blue, alpha) for the polymer line
POLYMER_BLUE = (0.125, 0.469, 0.707, 0.400)
TFACTORY_BLACK = (0.100, 0.050, 0.050, 0.700)

# Colour map, consisting of blue (n), orange (t), and purple (e)
MONOMER_COLOUR_MAP = {
    'n': (0.125, 0.469, 0.707, 0.700),
    't': (1.000, 0.498, 0.055, 0.700),
    'e': (0.593, 0.305, 0.637, 0.900)
    }


# Classes
#
# The elements from which we build a genome
NeutralElement = collections.namedtuple('NeutralElement', [])
TranscribedElement = collections.namedtuple('TranscribedElement', [])
EnhancerElement = collections.namedtuple('EnhancerElement', [])

# Additional enhancer elements
EnhancerAElement = collections.namedtuple('EnhancerElement', [])
EnhancerBElement = collections.namedtuple('EnhancerElement', [])


class LatticeGenome(object):
    """Self-avoiding polymer genome on lattice."""
    def __init__(self):
        """Set attributes to default values to make sure they exist."""
        # Types of monomer, NeutralElement, TranscribedElm, EnhancerElm.
        self.polymer = []
        # Auxilary set to keep track of enhancers (which have a different
        # energy contribution)
        self._chain_position_enhancer = set([])
        # Initial positions of monomers on the lattice.
        self.initial_positions = []
        self._positions_from_file = False

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
            for elm in conf['genome']:
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
                    self.polymer.append(EnhancerAElement())
                    try:
                        self.initial_positions.append(
                            [int(elm['x']), int(elm['y'])])
                    except KeyError:
                        pass
                    
                elif elm['type'] == 'enhancer_b':
                    self.polymer.append(EnhancerBElement())
                    try:
                        self.initial_positions.append(
                            [int(elm['x']), int(elm['y'])])
                    except KeyError:
                        pass

        except KeyError:
            print('EE Incorrect genome, perhaps unknown element')

        # I assume we have read all positions if the array is of equal length 
        # as the polymer one. Then, convert positions to np.array
        if len(self.polymer) == len(self.initial_positions):
            self.initial_positions = np.array(self.initial_positions)
            self._positions_from_file = True

        # Determine where enhancers are, store the indices.
        # AC: Using list comprehension to find indices of all enhancers
        self._chain_position_enhancer = set(
            [i for i,e in enumerate(self.polymer) 
                if type(e).__name__.lower().startswith('enhancer')])
        if options.verbose:
            print("# Found {} enhancer(s) in the genome.".format(
                len(self._chain_position_enhancer)))

    def json_encode(self):
        """Write genome to json dict."""
        # Ecrire les fichier de sortie dans le même format que les fichiers de
        # configuration, pq créer des collections.namedtuple si on l'utilise 
        # en tant que chaîne de caractères, et comment fonctionne la fonction 
        # type (renvoie type au lieu de TranscribedElement par ex).
        return {'genome': [{'type': type(e).__name__[0:-7].lower()} 
            for e in self.polymer]}

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
        Return for each monomer, all monomers that it forms a long range interaction with.
        """
        # return self._kdtree.query_ball_point(self.positions, 3, p=1.0)
        pass

    def attempt(self, positions):
        """
        Make a "move" on a copy of the polymer. Two possible moves are defined:
        the end-points can wiggle 90 degrees, and a turn can flip-flop.

        This is the so-called Mover set 1 (MS1) of Chan & Dill 1993, 1994.
        """
        aux_positions = np.copy(positions)
        # Pick a monomer
        idx = npr.randint(len(aux_positions))

        # End points wiggle
        if idx == 0:
            aux_positions[0] = self.__neighbour(aux_positions[1])
        elif idx == len(aux_positions) - 1:
            aux_positions[-1] = self.__neighbour(aux_positions[-2])
        else:
            # Is the monomer in a 90 degree turn?
            turn = (aux_positions[idx-1] + aux_positions[idx+1]) \
                    - 2 * aux_positions[idx]
            # Check if x and y are unequal to zero
            if np.all(turn != 0):
                aux_positions[idx] += turn

        return aux_positions

    def __neighbour(self, xy):
        """Return one of four possible neighbours."""
        offset = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        return xy + offset[npr.randint(len(offset))]

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
        self.radius = 3
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

    The main issue with using a lattice is that single monomer bonds are immutable -- unless I loosen that constraint --- which hampers the polymers flexibility. Are there alternative schemes that allow a more flexible polymer embedded on a lattice?
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
                self.interactions[(iaction['first'], iaction['second'])] = \
                    np.array(iaction['distance_to_energy'])
        except KeyError:
            print("WW Interactions not defined properly.")

    def json_encode(self):
        """Write simulation parameters to dict."""
        return {
            'end_time': self.end_time,
            'observe_time': self.stats_time,
            'interactions': [{
                "first": k[0], 
                "second": k[1], 
                "distance_to_energy": v.tolist()}
                for k, v in self.interactions.iteritems()]
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
                # A single simulation step is defined as to attempt to "move" each
                # element of the simulation. (At the moment only the genome.)
                self.step(time)
                time += 1

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
                    first = self.particles[pi].element_type(i - offset_i)
                    second = self.particles[pj].element_type(j - offset_j)

                    # Calculate energy of interaction
                    try:
                        energy += self.interactions[(first, second)][
                            np.sum(np.abs(p - aux_positions[j]))]
                    except KeyError:
                        energy += self.interactions[("neutral", "neutral")][
                            np.sum(np.abs(p - aux_positions[j]))]
                        
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

    def json_decode(self, conf):
        """Reading in which data to collect and save to file."""
        # At the moment we only track polymer positions
        try:
            self.polymer_positions_fname = \
                conf['observers']['polymer_positions']
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
            self._fig, self._axs = plt.subplots(nrows=2, ncols=2)

            # Axis 0: 2D polymer conformation
            self.__prepare_polymer_plot(time_step, world, xy, segments, tf)

            # Axis 1, 0: energy of the polymer over time
            self.__prepare_energy_plot(time_step, en)

            # Axis 0, 1: distance matrix
            self.__prepare_pairwise_distance_plot(time_step, 
                world.genome_positions())
            
            # Axis 1, 1:
            # self.__prepare_long_range_iaction_plot(time_step, world, nbs)

            # Connect key press and mouse button events to pause simulation.
            self._fig.canvas.mpl_connect("key_press_event", 
                self.__pause_simulation)
            self._fig.canvas.mpl_connect("button_press_event", 
                self.__pause_simulation)

            # Preparations done
            self._fig.show()

        else:
            # Update time and polymer positions
            self.__observe_polymer(time_step, xy, segments)
            # Update time vs energy graph
            self.__observe_energy(time_step, en)
            # Update distance matrix
            self.__observe_pairwise_distances(time_step, 
                world.genome_positions())
            # Update time and long range interactions
            # self.__observe_long_range_interactions(time_step, nbs)

            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

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
            transOffset=self._axs[0, 0].transData)
        trans = mpl.transforms.Affine2D().scale(self._fig.dpi/72.0)
        self._monomers.set_transform(trans)

        self._tfactory = mpl.collections.RegularPolyCollection(5, 
            sizes=[10.0 for _ in tf], offsets=tf,
            transOffset=self._axs[0, 0].transData)

        # Different monomers have different colours
        pt = world.genome.get_polymer_types_abbreviated()
        pc = [MONOMER_COLOUR_MAP[m] for m in pt]
        self._monomers.set_facecolor(pc)
        self._monomers.set_edgecolor(pc)

        # Transcription factory is black
        self._tfactory.set_facecolor(TFACTORY_BLACK)
        self._tfactory.set_edgecolor(TFACTORY_BLACK)

        self._axs[0, 0].add_collection(self._polymer_line)
        self._axs[0, 0].add_collection(self._monomers)
        self._axs[0, 0].add_collection(self._tfactory)

        # Make the plot pretty, no annoying tick or their labels
        lim = (-15, 15)
        self._axs[0, 0].set_xlim(*lim)
        self._axs[0, 0].set_ylim(*lim)
        self._axs[0, 0].set_xticks([])
        self._axs[0, 0].set_xticklabels([])
        self._axs[0, 0].set_yticks([])
        self._axs[0, 0].set_yticklabels([])
        # Make sure the plot is square
        self._axs[0, 0].set_aspect(1.0)

    def __observe_polymer(self, time_step, xy, segments):
        """Update time and polymer positions."""
        self._axs[0, 0].set_title('Time = {0}'.format(time_step))
        self._polymer_line.set_paths(segments)
        self._monomers.set_offsets(xy)        

    def __prepare_energy_plot(self, time_step, energy):
        """Prepare to plot energy over time."""
        # Create plotting area with 1 data point (at the moment)
        self._energy_line, = self._axs[1, 0].plot([time_step], [energy])

        # No extra spines
        self._axs[1, 0].spines['top'].set_visible(False)
        self._axs[1, 0].spines['right'].set_visible(False)

        # Add labels for clarity
        self._axs[1, 0].set_xlabel('Time (au)')
        self._axs[1, 0].set_ylabel('Energy (au)')

        # Keep track of current energy of the polymer
        self._energy_label = self._axs[1, 0].text(.9, .9, 
            'E = {0}'.format(energy), transform=self._axs[1, 0].transAxes)

    def __observe_energy(self, time_step, energy):
        """Update time vs energy graph."""
        # Label to keep track of the exact value
        self._energy_label.set_text('E = {0}'.format(energy))
        
        # It's not elegant to add a value to the plot data...
        aux_t, aux_e = self._energy_line.get_data()
        self._energy_line.set_data(
            (np.append(aux_t, time_step), np.append(aux_e, energy)))
        # Rescale etc. needed!
        self._axs[1, 0].relim()
        self._axs[1, 0].autoscale_view()

    def __prepare_pairwise_distance_plot(self, time_step, positions):
        """Prepare to plot distance matrices."""
        # Calculating distances, unfortunately computationally costly.
        d = spsp.distance.squareform(
                spsp.distance.pdist(positions, 'cityblock'))
        self._dist_img = self._axs[0, 1].imshow(d, interpolation='nearest')

        # Add labels for clarity
        self._axs[0, 1].set_title('Pairwise distance')
        self._axs[0, 1].set_ylabel('Monomer index')

        # Add colorbar, make sure to specify tick locations
        self._fig.colorbar(self._dist_img, ticks=[0, 5, 10, 15], 
            ax=self._axs[0, 1])

    def __observe_pairwise_distances(self, time_step, positions):
        """Update matrix of pairwise distances."""
        # Calculating distances, unfortunately computationally costly.
        d = spsp.distance.squareform(
                spsp.distance.pdist(positions, 'cityblock'))
        self._dist_img.set_data(d)

    def __prepare_long_range_iaction_plot(self, time_step, world, neighbours):
        """Prepare to plot many lines."""
        # How do I define a long range interaction? Any interaction with not-
        # immediate neighbours is of long(er) range. A first attempt is to 
        # sum the linear distances between monomers that engage in a long range
        # interaction.

        pt = world.genome.get_polymer_types_abbreviated()
        pc = [MONOMER_COLOUR_MAP[m] for m in pt]
        
        # i is the index to the current monomer, nbs is a list of 
        # neighbours (indices to monomers)
        self._lr_iaction_lines = []
        for i, nbs in enumerate(neighbours):
            line, = self._axs[1, 1].plot([time_step], 
                np.sum([np.abs(i - j) for j in nbs]), color=pc[i])
            self._lr_iaction_lines.append(line)
        
        # Add labels for clarity
        self._axs[1, 1].set_title('Long-range interactions')
        self._axs[1, 1].set_xlabel('Time (au)')
        self._axs[1, 1].set_ylabel('Sum of linear distances')

    def __observe_long_range_interactions(self, time_step, neighbours):
        """Calculate long range interactions and plot them per monomer."""
        for i, nbs in enumerate(neighbours):
            aux_t, aux_lria = self._lr_iaction_lines[i].get_data()
            self._lr_iaction_lines[i].set_data(
                (np.append(aux_t, time_step), 
                    np.append(aux_lria, np.sum([np.abs(i - j) for j in nbs]))))

        # Rescale etc. needed!
        self._axs[1, 1].relim()
        self._axs[1, 1].autoscale_view()


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
    out_data.update(world.genome.json_encode())

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

    # Set up simulation
    npr.seed(conf['random_seed'])

    # Build a genome
    g = LatticeGenome()
    g.json_decode(conf)
    
    # Some output to see what is going on
    out = "# Simulating a polymer genome of length {0}\n".format(len(g))
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
    time.sleep(60)

    # Write out simulation results
    write_simulation_results(options, conf, w, o)
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
