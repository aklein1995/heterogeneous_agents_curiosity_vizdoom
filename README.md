[//]: # (Image References)

[my_way_home_scenario]: https://user-images.githubusercontent.com/25618603/154808635-b430ab6c-be13-4e36-bdf6-62b366dc2a21.png "Environment"
[w0]:https://user-images.githubusercontent.com/25618603/154925438-0ca83e07-adbc-4407-b582-13c2a8a621df.gif "w0 gif"
[w1]:https://user-images.githubusercontent.com/25618603/154925553-728cb236-180d-42d7-867e-d542f933b726.gif "w1 gif"

# Collaborative Training of Heterogeneous Reinforcement Learning Agents in Environments with Sparse Rewards:What and When to Share?

In this paper we analyze how two independent agents that are deployed in the same environment (but at different instances of the latter) can learn faster when they do it in a Collaborative manner. For that purpose, both agents share knowledge in an online fashion without any previous expertise.

The key part in this work is that the agents are Heterogeneous, this is, they have different action spaces that allow one of the agents to have access to a bigger space domain. More importantly, that state space hinders better optimal solutions. Consequently, when sharing knowledge between agents negative transfer may arise.

The study is carried out at very sparse scenario, a modification of My Way Home scenario, where the agent is spawned at the bottom room and it has to reach to the goal, where the vest is located:

![Environment][my_way_home_scenario]

A unique reward of +1 is provided when arriving to the goal; 0 otherwise.

The state space is composed by first-view images. We process them to be 42x42 grayscale observations.

At this modification, a corridor (which is indeed a shortcut) has been added although is obstructed by a closed door (depicted with a white circle at the map).

Given this information, the agent has to learn how to best select actions. Four discrete actions are available for one of the agents, W1, corresponding to:
- **`0`** - do nothing.
- **`1`** - move forward.
- **`2`** - turn left.
- **`3`** - turn right.

Additionally, the other agent (W0) has another action that allows him to open doors (but does not report any advantage respect to do nothing if not necessary):
- **`4`** - open door.

Therefore, the work emerges on how to accelerate the training between both agents, taking into account that they will have different optimal solutions and some of the trajectories may well hinder undesired performances.

## Requirements
- Python 3.6
- [VizDooM](https://github.com/mwydmuch/ViZDoom) (check out dependencies)

### Dependencies

To set up the python environment to run the code in this repository and ensure all the used packages are installed, install the dependencies based on the requirements.txt:
```bash
pip3 -r install requirements.txt
```

## Basic Usage
You will find a config.conf file where all the different parameters are selected.

By default, only the analyzed environment is referred. If you want to use other .wad, just insert them into the wads folder and refer into the config.conf file.

Important to note that the .wad is called from .cfg file, where you can also modify other environment related parameters (i.e. actions).

## Results
The final policy of each worker when taking their optimal respective parts are next shown:

### Agent capable of open the door (W0)
![w0 gif][w0]
### Normal Agent (W1)
![w1 gif][w1]
