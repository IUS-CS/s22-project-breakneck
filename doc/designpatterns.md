  Currently, we still working on the Deeplearning models, I found out that we actually using Adpater pattern for our model classes.
All of our model classes are inherited from model wrapper(act as a adapter interface), the model wrapper provide serval functions that must be implemented
in the child class, once the child class implements it, it can be used anywhere else in our project, it's very similar to adapter pattern.  
  
  We may use Observer pattern and Singleton pattern in future.  
  
  Our next step is to continue working on our models. After that, we will start working on the GUI part of our project, this is where we may use observer pattern and singleton pattern for our project.
  We probably have to use callbacks to receive button click event(Observer pattern) and since we have only one Window GUI, we might put singleton pattern
  somewhere in the project.
