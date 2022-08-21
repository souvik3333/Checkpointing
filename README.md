# Checkpointing
Unified model state dict and architecture checkpoints utils.


Traditional model loading requires 
1. Model architecture
2. model state dict.

Normally we only save the model state dict in the checkpoint file, this forces us to
have the model architecture during inference. We can either load the model architecture
by instaling the package which contains the architecture dependency or we can copy the architecture file
to current repo. This creates additional dependencies for the production code.

One way to solve this problem is to save the model architecture to the checkpoint file itself and load both
the model architecture and state dict from the same checkpoint file. This solves few development issues:
1. If model architecture is changed in the architecture repo we wont have to use some old commit of the repo. 
For any architecture change and there is no need to map checkpoints to architecture version. 
2. Each checkpoint is self sufficient and independent.

Checkpointing add following utils:
1. Given a model architecture and state dict create a checkpointing checkpoint.
2. Given a checkpointing checkpoint return a the model with state dict loaded.
