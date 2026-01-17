# Notes
I originally started working on building these models in a fork of POLIANNA called ner_work, but eventually I wanted to do more experiments and the fork was too cumbersome, so I've picked up where I left off in this new repository.

### src 
src contains the pkl file and scripts from the original [POLIANNA github repository](https://github.com/kueddelmaier/POLIANNA) necessary to unpickle the dataset

### Experiments
I will be running four experiment categories, with a couple/multiple methods for each.
- a: One POLIANNA layer "InstrumentTypes" and 4/6 features from layer "PolicyDesignCharacteristics" (5)
- b: All POLIANNA layers (3)
- c: All [clean] POLIANNA features (10)
- d: One POLIANNA layer "InstrumentTypes" and 2/6 features from layer "PolicyDesignCharacteristics" (3)

### Methods
I am at least doing a single-head classification model and a multi-head classification model for each experiment type. We may try a prompting approach with dspy later but we want to check baselines first.