# SPH real-time Solver
An SPH particle-based fluid solver implemented on [Taichi Lang](https://www.taichi-lang.org) 

![Part_sim](./images/splash_300K.png)

## Dependencies
- Taichi Lang ( ``` pip install --upgrade taichi ```) 


## Install 
```git clone https://github.com/DimitriosApostolosPatsiouTerzidis/Thesis_taichi_fluid.git```

## Execution
To run the solver type in a terminal:
```python main.py```

## Simulation Configuration
Until a proper configuration file function is implemented, you can change physical and computational configuration setting by editing the values on the ```Constants.py``` file.

## TODO
- [ ] support configuration files for particle attributes (e.g. number of particles, radius), grid resolution
- [ ] add mesh to voxel import for complex geometry boundary interactions   
- [ ] boundary density and pressure correction (e.g. ghost particles or ...)
- [ ] support particle emitter 





