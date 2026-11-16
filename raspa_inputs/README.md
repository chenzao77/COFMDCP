# RASPA 2 methane inputs

These files define the COFMDCP methane GCMC setup at 298 K:

- TraPPE single-site methane (`methane.def`);
- UFF framework Lennard-Jones parameters;
- Lorentz-Berthelot mixing, 12.8 Å cutoff, and tail correction;
- pressure supplied in Pa and fugacity calculated by RASPA.

`python -m gcmc.gen_raspa` fills `simulation.input.template` and copies the force-field files into each run directory. The templates target RASPA 2.0.50.
