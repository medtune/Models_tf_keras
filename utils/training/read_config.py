import yaml

def decode(yamlFilename):
    """
    Returns dataset configuration, model configuration,
    and training configurations
    """
    with open(yamlFilename) as stream:
        config = yaml.load(stream)
    dataset_spec = config.get("dataset")
    train_spec = config.get("train")
    model_spec = config.get("model")
    model_spec = getModelSpec(dataset_spec, train_spec, model_spec)
    return dataset_spec, train_spec, model_spec

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
                "activation_func", "num_samples", "batch_size"]
    
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

def getInputFnSpec(dataSpec, trainSpec, modelSpec)
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
    pass
def getLabelNames(labelFile):
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