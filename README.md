# Reinforce Lab
This repository is meant to be a small where testing reinforcement learning algorithms.

### Authors/Maintainers

- Bernardo Fichera (bernardo.fichera@epfl.ch)

### Available environments
- particle
- unicycle (UNDER CONSTRUCTION)

### Available algorithms
- PPO (from https://github.com/higgsfield/RL-Adventure-2)

### Available controllers
- PID Controller

### Run examples
For the moment only the particle is the fully working example.

For training the model:
```sh
cd /src/example/particle
python target_train.py
```
If you have already trained the model and you want to retrain starting from the current weights configuration set the variable LOAD (line 16) to 'True'. From the same folder to run the simulation:
```sh
python simulate.py
```

### Documentation

UNDER CONSTRUCTION
