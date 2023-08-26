from dso import DeepSymbolicOptimizer

# Create and train the model
model = DeepSymbolicOptimizer("VisualCartPoleContinuous-v0.json")
# model = DeepSymbolicOptimizer("dso/dso/config/examples/control/CustomCartPoleContinuous-v0.json")
model.train()