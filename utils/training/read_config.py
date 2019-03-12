
import yaml

# Dictionnary that contain the default
# height and width for each model name
height_width = {
"densenet_121": (224,224),
"densenet_169": (224,224),
"densenet_201": (224,224),
"densenet_264": (224,224),
"inception_v3": (299,299),
"inception_resnet_v2": (299,299),
"mobilenet_v1": (224,224),
"mobilenet_v2": (224,224),
"nasnet_mobile": (224,224),
"nasnet_large": (331,331),
"resnet": (224,224),
"vgg_16": (224,224),
"vgg_19": (224,224),
}

channels = {
    "gray": 1,
    "rgb": 3,
    "rgba": 4
}

def get_default_shape(name):
    """
    Based on the model's name, the function returns
    a tuple of 2 dims representing the input shape
    """
    assert name in height_width.keys()
    return height_width.get(name)

def get_default_channels(imageType):
    """
    Based on the imageType (grayscale, RGB, RGBA), 
    the function returns a tuple of 3 dims 
    representing the input shape
    """
    assert imageType in channels.keys()
    return channels.get(imageType)

def decode(yamlFilename):
    """
    Returns dataset configurations, model configurations,
    and training configurations
    """
    with open(yamlFilename) as stream:
        config = yaml.load(stream)
    dataset_spec = config.get("dataset")
    train_spec = config.get("train")
    model_spec = config.get("model")
    model_spec = getModelFnSpec(dataset_spec, train_spec, model_spec)
    dataset_spec = getInputFnSpec(dataset_spec, train_spec, model_spec["name"])
    device_spec = train_spec

    return dataset_spec, model_spec, device_spec

def getModelFnSpec(dataSpec, trainSpec, modelSpec):
    """
    Having three dictionnaries: first one for dataset informations,
    the second one is for training specs, the third is about classification
    model spec. We want to complete it in order 
    to match the parameters needed for the AssembleModel class

    Argument:
        - dataSpec : dict
        - modelSpec : dict
    Returns:
        - dict representiong the model specifications as need by AssembleModel
    """
    tagsList = ["name", "image_type", "num_classes", "learning_rate",
                "activation_func", "num_samples", "batch_size", 
                "classification_layers", "classification_type"]
    
    dataSpecKeys = dataSpec.keys()
    trainSpecKeys = trainSpec.keys()

    for tag in tagsList:
        if tag not in modelSpec:
            if tag in dataSpecKeys:
                modelSpec[tag] = dataSpec[tag]
            elif tag in trainSpecKeys:
                modelSpec[tag] = trainSpec[tag]
            else:
                raise KeyError('The requested key %s is not found in both training\
                                and dataset specifications. PLease refer to the yaml file\
                                configuration'%(tag))
    return modelSpec

def getInputFnSpec(dataSpec, trainSpec, modelName):
    """
    Having three dictionnaries: first one for dataset informations,
    the second one is for training specs, the third is about classification
    model spec. We want to complete it in order 
    to match the parameters needed for the input_fn function

    Argument:
        - dataSpec : dict
        - modelSpec : dict
    Returns:
        - dict representing the model specifications as need by AssembleModel
    """
    tagsList = ["dataset_dir","file_pattern","image_type", "num_classes", "batch_size",
                "num_epochs", "num_samples", "shuffle_buffer_size"]
    
    trainSpecKeys = trainSpec.keys()
    for tag in tagsList:
        if tag not in dataSpec.keys():
            if tag in trainSpecKeys:
                dataSpec[tag] = trainSpec.pop(tag)
            else:
                raise KeyError('The requested key %s is not found in both training\
                                and dataset specifications. Please refer to the yaml file\
                                configuration'%(tag))
    dataSpec["image_size"] = get_default_shape(modelName)
    dataSpec["image_channels"] = get_default_channels(dataSpec["image_type"])
    return dataSpec

def namesToLabels(labelFile):
    """
    Given a label file, we extact two dictionnaries:
        - first maps each label name (string) to an integer value
        - The second maps each integer value to a string value
    """
    with open(labelFile) as f:
        labels = f.readlines()
    # Dict mapping label names to integer values :
    namesToInt = dict.fromkeys(labels, "und")
    # Dict mapping integer to strings : 
    intList = [i for i in range(len(labels))]
    intToNames = dict(zip(intList, labels))
    return namesToInt, intToNames

def setDeviceConfig(distributionStrategy, xlaStrategy):
    """
    Arguments:
        - distributionStrategy : boolean. If true, enables 
    MirroredStrategy
        - xlaStrategy : boolean. If true, enables xla computation
    optimization
    """
    import tensorflow as tf
    strategy, jitLevel = None, 0
    if distributionStrategy:
        strategy = tf.contrib.distribute.MirroredStrategy()
    if xlaStrategy:
        jitLevel = tf.OptimizerOptions.ON_1
    # Define tf.ConfigProto() as config to pass to estimator config:
    config = tf.ConfigProto()
    # NOTE: The following line is added for Nvidia RTX
    # (https://github.com/tensorflow/tensorflow/issues/24496)
    # config.gpu_options.allow_growth = True
    #Define optimizers options based on jit_level:
    config.graph_options.optimizer_options.global_jit_level = jitLevel
    return strategy, config