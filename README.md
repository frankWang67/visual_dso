# visual_dso
Archive of codes in the summer research of tong class in 2023: Reinforcement Learning Based on Neuron-Symbol.

This is an extension of the repo `dso` (deep symbolic optimization, https://github.com/dso-org/deep-symbolic-optimization) with frames as input.
The part I added is in `./dso/dso/vision_module` and `./dso/dso/task/control/visual_control.py`.

All usage is the same as `dso`, except for you can use frames as input by changing `task_type` into `vision_control` in the config `json` file. Here's an example `VisualCartPoleContinuous-v0.json`:

```json
{
   "task" : {
      "task_type" : "vision_control",
      "env" : "CustomCartPoleContinuous-v0",
      "action_spec" : [null],
      "n_episodes_train" : 5,
      "n_episodes_test" : 10,
      "success_score": 800.0,
      "function_set" : ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", 0.1, 1, 5],
      "fix_seeds" : false,
      "episode_seed_shift" : 0,
      "env_kwargs" : {"dt" : 0.02},
      "reward_scale" : true
   },
   "training" : {
      "n_samples" : 100000,
      "batch_size" : 20,
      "n_cores_batch": 2
   },
   "policy" : {
      "policy_type" : "rnn", 

      "max_length" : 100,

      "cell" : "lstm",
      "num_layers" : 1,
      "num_units" : 32,
      "initializer" : "zeros"
   },
   "prior" : {
      "soft_length" : {
         "loc" : 50,
         "scale" : 5,
         "on" : true
      }
   }
}
```

The observation through frames consists of 3 frames of object detected, with their positions and angles (in a 2D scene) concatenated as an array.  
Codes are in the `master` branch.
