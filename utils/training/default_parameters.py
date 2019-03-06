# Dictionnary that contain the default
# height and width for each model name
height_width = {
"densenet121": (224,224),
"densenet169": (224,224),
"densenet201": (224,224),
"densenet264": (224,224),
"inceptionv3": (299,299),
"inception_resnet_v2": (299,299),
"mobilenet": (224,224),
"mobilenetv2": (224,224),
"nasnet_mobile": (224,224),
"nasnet_large": (331,331),
"resnet": (224,224),
"vgg16": (224,224),
"vgg19": (224,224),
"xception": (299,299)
}

channels = {
    "gray": (1,),
    "rgb": (3,),
    "rgba": (4,)
}

def get_default_shape(name):
    """
    Based on the model's name and the image type
    (grayscale, RGB, RGBA), the function returns
    a tuple of 3 dims representing the input shape
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