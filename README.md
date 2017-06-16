# LATTICE GENOME

I am designing and implementing lattice genomes for exploratory studies into genome folding. As lattice genomes are computationally quite light, they are perfect for use in education (i.e. on older computers and not-so-powerful laptops).


# EXAMPLES

  python lattice_genome.py -c simple_config.json


# LOOPS

The variant 'lattice_genome_loops.py' enables the use of enhancer monomers to explore loop formation via enhancer interactions.


# FUTURE WORK

Also known as the to-do list:

- Move positions from LatticeGenome to World.
- Move step and attempt from LatticeGenome to World.
- Move energy calculation from LatticeGenome to World.

- Create World matrix to define the pairwise energy of elements.
- Create transcription factory object.
- Adjust step and attempt to include interactions between the factory and the
  polymer chain.

- Observers should be split up in different observers.
