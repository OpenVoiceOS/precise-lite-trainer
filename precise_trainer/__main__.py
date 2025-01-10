import click

from precise_trainer import PreciseTrainer


@click.group()
def cli():
    """OpenVoiceOS Precise Trainer CLI"""
    pass


@cli.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), default=None, help="Path to save the converted model.")
def convert(model_file, output):
    """Convert a model to TFLite format."""
    output = output or model_file + ".tflite"
    PreciseTrainer.convert(model_file, output)
    click.echo(f"Converted model saved to {output}")


@cli.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.argument('dataset_folder', type=click.Path(exists=True))
def test(model_file, dataset_folder):
    """Test a model on a dataset."""
    PreciseTrainer.test_from_file(model_file, dataset_folder)


@cli.command()
@click.argument('model_name', type=str)
@click.argument('dataset_folder', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
@click.option('--log_dir', type=click.Path(), default=None, help="Directory for TensorBoard logs.")
@click.option('--epochs', type=int, default=100, help="Number of training epochs.")
def train(model_name, dataset_folder, model_path, log_dir, epochs):
    """Train a model."""
    trainer = PreciseTrainer(model_path, dataset_folder, epochs=epochs, log_dir=log_dir)
    model_file = trainer.train()
    click.echo(f"Model trained and saved to {model_file}")
    trainer.test()


@cli.command()
@click.argument('model_name', type=str)
@click.argument('dataset_folder', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
@click.option('--log_dir', type=click.Path(), default=None, help="Directory for TensorBoard logs.")
@click.option('--epochs', type=int, default=100, help="Number of training epochs.")
@click.option('--mini_epochs', type=int, default=10, help="Number of mini-epochs per replacement.")
def train_with_replacement(model_name, dataset_folder, model_path, log_dir, epochs, mini_epochs):
    """Train a model with data replacement."""
    trainer = PreciseTrainer(model_path, dataset_folder, epochs=epochs, log_dir=log_dir)
    model_file = trainer.train_with_replacement(mini_epochs=mini_epochs)
    click.echo(f"Model trained with replacement and saved to {model_file}")
    trainer.test()


@cli.command()
@click.argument('model_name', type=str)
@click.argument('dataset_folder', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
@click.option('--log_dir', type=click.Path(), default=None, help="Directory for TensorBoard logs.")
@click.option('--epochs', type=int, default=100, help="Number of training epochs.")
@click.option('--mini_epochs', type=int, default=20, help="Number of mini-epochs for incremental training.")
def train_incremental(model_name, dataset_folder, model_path, log_dir, epochs, mini_epochs):
    """Train a model incrementally."""
    trainer = PreciseTrainer(model_path, dataset_folder, epochs=epochs, log_dir=log_dir)
    model_file = trainer.train_incremental(mini_epochs=mini_epochs)
    click.echo(f"Incrementally trained model saved to {model_file}")
    trainer.test()


@cli.command()
@click.argument('model_name', type=str)
@click.argument('dataset_folder', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
@click.option('--log_dir', type=click.Path(), default=None, help="Directory for TensorBoard logs.")
@click.option('--epochs', type=int, default=100, help="Number of training epochs.")
@click.option('--cycles', type=int, default=20, help="Number of optimization cycles.")
def train_optimized(model_name, dataset_folder, model_path, log_dir, epochs, cycles):
    """Train a model with hyperparameter optimization."""
    trainer = PreciseTrainer(model_path, dataset_folder, epochs=epochs, log_dir=log_dir)
    model_file = trainer.train_optimized(cycles=cycles)
    click.echo(f"Optimized model saved to {model_file}")
    trainer.test()


if __name__ == '__main__':
    cli()
