# traffic light detection
This is for udacity capstone project. A traffic light detector using [squeezeNet](https://arxiv.org/abs/1602.07360).

The model (model.h5) is generated using [model.save(filepath)](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model). It saves the whole model (architecture + weights + optimizer state). The saved model can be loaded by keras.models.load_model(filepath).

