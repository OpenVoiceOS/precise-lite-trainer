# OpenVoiceOS precise trainer

WIP - open during construction

Several training strategies are available, each may provide better results for different datasets and wake words, some sounds might be easier to learn than others and the kinds of data available for each word will be different

- train - select epochs, batch size and go!
- train with replacement - use a different subset of the training data in every epoch, helps avoid overfitting
- train incremental - every epoch test the model and move false positives to training set, helps if you have an unbalanced dataset (a lot not-ww samples)
- train incremental with replacement - unbalanced dataset + overfitting to specific voices
- train optimized - using [bbopt](https://github.com/evhub/bbopt) search the optimal hyperparams (dropout and recurrent units), train several models and keep best one
- train optimized incremental
- train optimized with replacement

```python
from precise_trainer import PreciseTrainer
from precise_trainer.model import ModelParams

extra_metrics = False
no_validation = False
freeze_till = 0
sensitivity = 0.2

params = ModelParams(skip_acc=no_validation, extra_metrics=extra_metrics,
                     loss_bias=1.0 - sensitivity, freeze_till=freeze_till)
model_name = "hey_computer"
folder = f"/home/user/ww_datasets/{model_name}"
model_path = f"/home/user/trained_models/{model_name}"
log_dir = f"logs/fit/{model_name}"

trainer = PreciseTrainer(model_path, folder, epochs=100, log_dir=log_dir)

# pick one training method
model_file = trainer.train()
model_file = trainer.train_with_replacement(mini_epochs=10)
model_file = trainer.train_incremental(mini_epochs=20)
model_file = trainer.train_incremental_with_replacement(balanced=True, porportion=0.6)
model_file = trainer.train_optimized(cycles=20)
model_file = trainer.train_optimized_with_replacement(porportion=0.8)
model_file = trainer.train_optimized_incremental(cycles=50)

# convert a previous model
model_file = ".../my_model"
PreciseTrainer.convert(model_file, model_file + ".tflite")

# test a previous model
model_file = ".../my_model.tflite"
PreciseTrainer.test(model_file, folder)
```