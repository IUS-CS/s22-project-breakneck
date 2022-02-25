# File Structure 
Under **/src/** we have several important sub-directories.
1. **/src/data/** This folder contains all training datasets (100,000 images).
2. **/src/models/** This folder is where we put all model related stuffs (for example: model training script, model testing script and model codes).
3. **/src/models/saved_models/** This folder is where all of our model weights saved at.
4. **/src/program_builder/** This folder conatins program build scripts.

# Project Architecture
Our project is divided into two parts GUI and model

**GUI**  
we haven't started yet.

**Model**  
We define each model as a class and all model classes are inherited from a model wrapper class.
This model wrapper class have two functions that must be implemented in the child class(we want to do something like abstract class in Java),
  - **def _define_model(self)** in this function we have to build our keras model and returns it
  - **def _compile_model(self, model)** in this function we have to compile our keras model

This model wrapper class provide some useful features such as auto load saved weights, auto save model weights while training etc...  
so once our child class implemented these two functions, we got features from model wrapper without adding any extra code and more importantly we now have a model standard, 
it would be very helpful in future development.
