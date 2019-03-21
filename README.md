# Models_tf_keras

Models_tf_keras is a Python library built on top of Tensorflow [tensorflow](tensorflow.org).
The starting point of the program is a configuration file, located in "/yaml" folder, under
the name of "config.yaml"

## Installation


Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install requirement.txt
```

## Usage
```yaml
dataset:
  dataset_dir: 
  file_pattern: 
  num_samples: 
  num_classes: 
  image_type: 
model:
  name: 
  classification_layers: 
  classification_type: 
  optimizer_noun: 
  activation_func: 
train:
  distribute: 
  xla : 
  batch_size: 
  learning_rate:
    initial : 
    decay_factor : 
    before_decay : 
  num_epochs: 
  shuffle_buffer_size:
```

## Contributing
Coming soon

## License
[MIT](https://choosealicense.com/licenses/mit/)