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

# Colour map, consisting of blue (n) and orange (t), both with 60% transparency
MONOMER_COLOUR_MAP = {
    'n': (0.125, 0.469, 0.707, 0.600),
    't': (1.000, 0.498, 0.055, 0.600),
    'e': (0.5, 0.798, 0.055, 0.600)
    }


# Classes
#
# The elements from which we build a genome
NeutralElement = collections.namedtuple('NeutralElement', [])
TranscribedElement = collections.namedtuple('TranscribedElement', [])
EnhancerElement = collections.namedtuple('EnhancerElement', [])


class LatticeGenome(object):
    """Self-avoiding polymer genome on lattice."""
    def __init__(self):
        """Set attributes to default values to make sure they exist."""
        # Types of monomer, NeutralElement or TranscribedElement.
        self._polymer = []
        # Positions of monomers on the lattice
        # Positions initiales déterminées à partir du fichier de configuration T/F
        self._positions = []
        self._positions_from_file = False
        # Energy accumulated due to non-adjacent monomers being neighbours
        self._energy = 0.0
        # Temperature of world (arguably an attribute of the world, not genome)
        self._temp = 0.0
        # Binary tree that keeps track of monomers in space. Used to quickly 
        # look up neighbours of a given monomer. About 4--5 times faster than
        # naive all-against-all calculation.
        self._kdtree = None

    def json_decode(self, conf):
        """Build genome from json dict."""
        # Reset polymer and positions to be empty
        # a quoi cela sert-il ? Les listes ne sont-elles pas déjà vides juste après la création de l'objet ?
        self._polymer = []
        self._positions = []

        try:
            for elm in conf['genome']:
                # la clef 'genome' a pour valeur une liste de dictionnaire constitués de 3 clefs (type, x, y)
                # Chaque élément à un type et peut avoir un positions xy si l'on utilise un fichier de config issu d'une simulation précedente.
                if elm['type'] == 'neutral':
                    self._polymer.append(NeutralElement())
                    try:
                        self._positions.append(
                            [int(elm['x']), int(elm['y'])])
                # Si le fichier de config ne contient pas de position xy, le programme continue. Elle seront générées aléatoirement par la suite.
                    except KeyError:
                        pass

                elif elm['type'] == 'transcribed':
                    self._polymer.append(TranscribedElement())
                    try:
                        self._positions.append(
                            [int(elm['x']), int(elm['y'])])
                    except KeyError:
                        pass
                    
                elif elm['type'] == 'enhancer' :
                    self._polymer.append(EnhancerElement())
                    try:
                        self._positions.append(
                            [int(elm['x']), int(elm['y'])])
                    except KeyError:
                        pass
                                

        except KeyError:
            print('EE Incorrect genome, perhaps unknown element')

        # I assume we have read all positions if the array is of equal length 
        # as the polymer one. Then, convert positions to np.array
        if len(self._polymer) == len(self._positions):
            self._positions = np.array(self._positions)
            self._positions_from_file = True

        try:
            #clef 'temperature' contient une valeur de température
            self._temp = conf['temperature']
        except KeyError:
            print('EE Missing temperature.')

    def json_encode(self):
        """Write genome to json dict."""
        #Ecrire les fichier de sortie dans le même format que les fichiers de configuration
        #pq créer des collections.namedtuple si on l'utilise en tant que chaîne de caractères, et comment fonctionne la fonction type (renvoie type au lieu de TranscribedElement par ex)
        result = {}
        result['genome'] = [
            {'type': type(e).__name__[:-7].lower(), 'x': p[0], 'y': p[1]}
                for e,p in zip(self._polymer, self._positions)]
        result['temperature'] = self._temp
        return result


    def get_polymer_types_abbreviated(self):
        """
        Return polymer as a list of 'n's and 't's, respectively neutral and 
        transcribed monomers.
        """
        return [type(e).__name__[0].lower() for e in self._polymer]

    def get_positions(self):
        """Return the positions of the monomers (x, y)."""
        return self._positions

    def get_energy(self):
        """Return the energy of the polymer."""
        return self._energy

    def has_positions_from_file(self):
        return self._positions_from_file

    def initial_positions(self):
        """
        Set initial position of each of the monomers. Default is a random 
        configuration.
        """
        def __next_position(xy):
            """Auxilary function to choose a step (n,s,e) on the lattice."""
            #la fonction ne choisit pas les mêmes directions selon la direction choisit à la position précédente, ce sont bien 3 positions différentes mais pas les mêmes à chaque fois
            x, y = tuple(xy)
            nbh = [[-y, x], [x, y], [y, -x]]
            return nbh[npr.randint(3)]

        # First monomer is positioned at [0,0], second [x,y] takes one step 
        # north, south, or east on the lattice.
        x = npr.randint(2)
        y = npr.choice([-1, 1]) if x == 0 else 0
        
        # Two positions done, let's do the rest
        self._positions = [[0, 0], [x, y]]
        for _ in self._polymer[2:]:
            self._positions.append(__next_position(self._positions[-1]))

        # Input to cumsum is a list, returns a numpy array
        self._positions = np.cumsum(self._positions, axis=0)

    def initial_energy(self):
        """
        Energy of a polymer is sum of contact energy between not-nearest
        neighbours.
        """
        self._energy, self._kdtree = self.__calculate_energy(self._positions)

    def step(self):
        """
        Accept attempt if it lowers the energy or simply by chance using 
        an exponential distribution.
        """
        new_positions, new_energy, new_tree = self.__attempt()
        delta_e = new_energy - self._energy
        if delta_e < 0.0 or npr.random() <= np.exp(-(delta_e / self._temp)):
            # Biased acceptance of attempts
            self._positions = new_positions
            self._energy = new_energy
            self._kdtree = new_tree

    def translate_to_origin(self):
        """
        Calculate centre of mass and shift all positions to have centre at 
        the origin (0,0).
        """
        # Watch out, integer division!
        centroid = np.sum(self._positions, axis=0) // len(self._positions)
        # Translate all points.
        self._positions -= centroid

    def __attempt(self):
        """
        Make a "move" on a copy of the polymer. Two possible moves are defined:
        the end-points can wiggle 90 degrees, and a turn can flip-flop.

        This is the so-called Mover set 1 (MS1) of Chan & Dill 1993, 1994.
        """
        positions = np.copy(self._positions)
        # Pick a monomer
        idx = npr.randint(len(positions))

        # End points wiggle
        if idx == 0:
            positions[0] = self.__neighbour(positions[1])
        elif idx == len(positions) - 1:
            positions[-1] = self.__neighbour(positions[-2])
        else:
            # Is the monomer in a 90 degree turn?
            turn = (positions[idx-1] + positions[idx+1]) - 2 * positions[idx]
            # Check if x and y are unequal to zero
            if np.all(turn != 0):
                positions[idx] += turn

        energy, tree = self.__calculate_energy(positions)
        return positions, energy, tree

    def __neighbour(self, xy):
        """Return one of four possible neighbours."""
        offset = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        return xy + offset[npr.randint(len(offset))]

    def __calculate_energy(self, aux_positions):
        """
        Calculate energy. 

        A self-avoiding polymer defines an energy 'penalty' between 
        non-adjacent monomers that touch each other. Here I define three 
        energy levels:

        - 10: monomers are on top of each other.
        - 1 : monomers are next to each other.
        - 0.1: monomers are diagonally close.
        - -10: monomères superposés de 2 enhancers
        - -1 : monomères proches de 2 enhancers
        - -0.1: monomères éloignés de 2 enhancers

        These energy levels should lead to polymers that become elongated.
        """
        DIST_TO_ENERGY_others = [10, 1, 0.1]
        DIST_TO_ENERGY_enhancers = [-10, -1, -0.1]

        # Start with zero energy.
        energy = 0
        
        # KDTree expects all positions of the monomers, a max distance (2), and
        # a distance measure (1.0 = Manhattan distance).
        kdtree = spsp.cKDTree(aux_positions, compact_nodes=False, 
            balanced_tree=False)
        neighbours = kdtree.query_ball_point(aux_positions, 2, p=1.0)
        
        chain_position_enhancer=[]

        for i in range(len(self._polymer)):
            if type(self._polymer[i])==type(EnhancerElement()):
                chain_position_enhancer.append(i) 
    
        for i, pnbs in enumerate(zip(aux_positions,neighbours)):
            p, nbs= pnbs
            if i in chain_position_enhancer:
                for j in nbs:
                    if j!=i:
                        if j in chain_position_enhancer:
                            energy+=DIST_TO_ENERGY_enhancers[np.sum(np.abs(p - aux_positions[j]))]
                        elif j not in chain_position_enhancer:
                            energy+=DIST_TO_ENERGY_others[np.sum(np.abs(p - aux_positions[j]))]
            else:
                for j in nbs:
                    if j != i:
                        energy+=DIST_TO_ENERGY_others[np.sum(np.abs(p - aux_positions[j]))]

        return energy, kdtree

    def __len__(self):
        return len(self._polymer)

    def __str__(self):
        return "-".join(self.get_polymer_types_abbreviated())


class World(object):
    """
    A simplified nuclear environment. Currently, the world consists of two
    objects, namely the genome and a transcription factory.
    """
    def __init__(self):
        self._end_time = 100
        self._stats_time = 10
        self._translate_time = np.max([1000, int(self._end_time / 100)])

        self._genome = []
        # self._transcription_factory = []

    def json_decode(self, conf):
        """Read in simulation parameters."""
        try:
            self._end_time = conf['end_time']
            self._stats_time = conf['observe_time']
        except KeyError:
            print("EE Missing time.")

    def json_encode(self):
        """Write simulation parameters to dict."""
        return {
            'end_time': self._end_time,
            'observe_time': self._stats_time
            }

    def add_genome(self, genome):
        self._genome = genome
        if not self._genome.has_positions_from_file():
            self._genome.initial_positions()
        self._genome.initial_energy()

    def get_genome(self):
        return self._genome

    def simulate(self, observers):
        """Monte Carlo simulation algorithm."""
        time = 0
        while time < self._end_time:
            # Output statistics
            if time % self._stats_time == 0:
                observers.observe(time, self)
            
            # A single simulation step is defined as to attempt to "move" each
            # element of the simulation. (At the moment only the genome.)
            self.__step(time)
            time += 1

    def __step(self, time):
        """Attempt to change world..."""
        self._genome.step()
        if time % self._translate_time == 0:
            self._genome.translate_to_origin()
        # self._transcription_factory.step()


class Observers(object):
    """Observer manager."""
    def __init__(self):
        self._first = True
        self._fig = None
        
        self._polymer_line = None
        self._monomers = None

        self._energy_line = None
        self._energy_label = None

    def json_decode(self, conf):
        """At the moment there is no configuration to read in."""
        pass

    def json_encode(self):
        """There is also nothing to write out."""
        pass

    def observe(self, time_step, world):
        """
        Connect observers to the right parts of the world and let them compute 
        their statistics.
        """
        # Get polymer data, first energy then positions
        en = world.get_genome().get_energy()

        # Reshape into sequence of line segments [[(x0,y0),(x1,y1)],...]
        xy = world.get_genome().get_positions()
        xy = xy.reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])

        if self._first:
            # Set up visualization only once
            self._first = False
            self._fig, self._axs = plt.subplots(nrows=2, ncols=1)

            # Axis 0: 2D polymer conformation
            self.__prepare_polymer_plot(time_step, world, xy, segments)

            # Axis 1: energy of the polymer over time
            self.__prepare_energy_plot(time_step, en)

            # Preparations done
            self._fig.show()

        else:
            # Updating time and polymer positions
            self.__observe_polymer(time_step, xy, segments)
            # Updating time vs energy graph
            self.__observe_energy(time_step, en)

            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def __prepare_polymer_plot(self, time_step, world, xy, segments):
        """Prepare polymer line and monomers."""
        # The polymer with its monomers is built from lines and polygons
        self._polymer_line = mpl.collections.LineCollection(segments)
        self._polymer_line.set_color(POLYMER_BLUE)

        self._monomers = mpl.collections.RegularPolyCollection(4, 
            sizes=[10.0 for _ in xy], offsets=xy, 
            transOffset=self._axs[0].transData)
        trans = mpl.transforms.Affine2D().scale(self._fig.dpi/72.0)
        self._monomers.set_transform(trans)

        # Different monomers have different colours
        pt = world.get_genome().get_polymer_types_abbreviated()
        pc = [MONOMER_COLOUR_MAP[m] for m in pt]
        self._monomers.set_facecolor(pc)
        self._monomers.set_edgecolor(pc)

        self._axs[0].add_collection(self._polymer_line)
        self._axs[0].add_collection(self._monomers)

        # Make the plot pretty, no annoying tick or their labels
        lim = (-15, 15)
        self._axs[0].set_xlim(*lim)
        self._axs[0].set_ylim(*lim)
        self._axs[0].set_xticks([])
        self._axs[0].set_xticklabels([])
        self._axs[0].set_yticks([])
        self._axs[0].set_yticklabels([])
        # Make sure the plot is square
        self._axs[0].set_aspect(1.0)

    def __observe_polymer(self, time_step, xy, segments):
        """Update time and polymer positions."""
        self._axs[0].set_title('Time = {0}'.format(time_step))
        self._polymer_line.set_paths(segments)
        self._monomers.set_offsets(xy)        

    def __prepare_energy_plot(self, time_step, energy):
        """Prepare to plot energy over time."""
        # Create plotting area with 1 data point (at the moment)
        self._energy_line, = self._axs[1].plot([time_step], [energy])

        # No extra spines
        self._axs[1].spines['top'].set_visible(False)
        self._axs[1].spines['right'].set_visible(False)

        # Add labels for clarity
        self._axs[1].set_xlabel('Time (au)')
        self._axs[1].set_ylabel('Energy (au)')

        # Keep track of current energy of the polymer
        self._energy_label = self._axs[1].text(.9, .9, 
            'E = {0}'.format(energy), transform=self._axs[1].transAxes)        

    def __observe_energy(self, time_step, energy):
        """Update time vs energy graph."""
        # Label to keep track of the exact value
        self._energy_label.set_text('E = {0}'.format(energy))
        
        # It's not elegant to add a value to the plot data...
        aux_t, aux_e = self._energy_line.get_data()
        self._energy_line.set_data(
            (np.append(aux_t, time_step), np.append(aux_e, energy)))
        # Rescale etc. needed!
        self._axs[1].relim()
        self._axs[1].autoscale_view()


# Functions
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


def write_json_configuration(fname, out_data):
    """Write out simulation parameters and results (in JSON format)."""
    try:
        outfile = open(fname, 'w')
    except IOError:
        print('EE Cannot open for writing:', fname)
    else:
        conf = outfile.write(
            json.dumps(out_data, indent=2, encoding='utf-8').decode('utf8'))


def write_simulation_results(opt, config, world, observers):
    """
    Write out configuration and simulation results. The output file can be 
    used to continue a simulation at a later point.
    """
    out_data = {'random_seed': config['random_seed']}
    out_data.update(world.json_encode())
    out_data.update(world.get_genome().json_encode())
    # Ignoring observers (for now)

    # Preparing file name
    outfilename = os.path.expanduser(opt.save)
    print("# Writing results to", outfilename)
    write_json_configuration(outfilename, out_data)


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
    # Lire le fichier de configuration
    conf = read_json_configuration(options.config)

    # Set up simulation
    # Création d'une unique série de nombre aléatoire à partir d'un premier chiffre
    # donné dans le fichier de configuration 
    npr.seed(conf['random_seed'])

    # Build a genome
    # Créer un objet de la classe LatticeGenome
    g = LatticeGenome()
    g.json_decode(conf)
    
    # Some output to see what is going on
    out = "# Simulating a polymer genome of length {0}\n".format(len(g))
    print(out)

    # Build the molecular world in which the genome is placed
    # Créer un objet de la classe world
    w = World()
    w.json_decode(conf)
    w.add_genome(g)

    # Set up which statistics to compute during simulation
    o = Observers()
    o.json_decode(conf)
    
    # Run the simulation
    w.simulate(o)
    # If you do not want the plot window to close immediately, uncomment 
    # the two lines below.
    print("# Giving you some time to enjoy the plots...")
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
            help='results file name')
        (options, args) = parser.parse_args()
        
        #if len(args) < 1:
        #    parser.error('missing argument')

        if options.verbose: print(time.asctime())
        main()
        if options.verbose: print(time.asctime())
        if options.verbose: print('# total time (min):', end='')
        if options.verbose: print((time.time() - start_time) / 60.0)
        sys.exit(0)

    except KeyboardInterrupt, e: # Ctrl-C
        raise e

    except SystemExit, e: # sys.exit()
        raise e

    except Exception, e:
        print('Error, unexpected exception')
        print(str(e))
        traceback.print_exc()
        os._exit(1)
