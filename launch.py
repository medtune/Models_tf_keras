import tensorflow as tf
import os
import sys
import utils.training.read_config as read_config
import models.cnn.base as base
import inputs.images.dataset_images as dataset_images

def main():
    # Open and read the yaml file :
    cwd = os.getcwd()
    # Yaml filename
    yamlFilename = os.path.join(cwd, "yaml","config.yaml")
    # Use the config file and extract dataset, model and training specs
    datasetSpec, modelSpec, deviceSpec  = read_config.decode(yamlFilename)
    # Construct the training folder 
    jobDir = os.path.join(cwd, "job_"+ modelSpec.get("name"))
    # Create log_dir : argscope_config
    if not os.path.exists(jobDir):
        os.mkdir(jobDir)
    #===================================================================== Training ===========================================================================#
    #Adding the graph:
    #Set the verbosity to INFO level
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # Define strategy training variable, xla computations : 
    # distribute::distribution Strategy ; xla:: xla computation optimization
    strategy, config = read_config.setDeviceConfig(deviceSpec["distribute"],
                                                    deviceSpec["xla"]  )
    model = base.AssembleComputerVisionModel(modelSpec)
    
    # Define configuration:
    run_config = tf.estimator.RunConfig(save_checkpoints_steps = model.num_batches_per_epoch,
                                        keep_checkpoint_max=datasetSpec["num_epochs"],
                                        train_distribute=strategy,
                                        eval_distribute=strategy,
                                        session_config=config)
    # Define warm start setting using initModel method:
    warmStartSetting = model.initModel(jobDir)
    estimator = tf.estimator.Estimator(model.model_fn,
                                       model_dir=jobDir,
                                       config=run_config,
                                       warm_start_from=warmStartSetting)
    
    #Define trainspec estimator, including max number of step for training
    max_step = model.num_batches_per_epoch * datasetSpec["num_epochs"]
    train_spec = tf.estimator.TrainSpec(input_fn = dataset_images.\
                                        get_input_fn(tf.estimator.ModeKeys.TRAIN, datasetSpec), 
                                        max_steps = max_step)
    #Define evalspec estimator
    eval_spec = tf.estimator.EvalSpec(input_fn=dataset_images.\
                                      get_input_fn(tf.estimator.ModeKeys.EVAL, datasetSpec))       
    #Run the training and evaluation (1 eval/epoch)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    main()