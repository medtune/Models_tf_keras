import tensorflow as tf
import os
import sys
import urllib
import tarfile
import zipfile

"""
It also defines training hooks for monitoring hardware performance and
time processing
It also provides utily functions to call tf.summary.xxx depending on the
model
"""

def _print_download_progress(count, block_size, total_size):
    """
    Function used for printing the download progress.
    Used as a call-back function in maybe_download_and_extract().
    """

    # Percentage completion.
    pct_complete = float(count * block_size) / total_size

    # Limit it because rounding errors may cause it to exceed 100%.
    pct_complete = min(1.0, pct_complete)

    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

def download_imagenet_checkpoints(checkpointName, url, downloadDir):
    """
    Given the name of the model, we first extract the
    Imagenet URL weights from checkpoints dicts (.ref: famous_cnn)
    
    # Arguments:
        - checkpointName
        - url : Correspond to the url to tar/zip file
        - downloadDir : Correspond to jobPath/imagenet_weights
    """
    fileExtension = ".tar.gz"
    ckptExtension = ['.ckpt.meta', '.ckpt.index', '.ckpt.data-00000-of-00001']
    
    fileName = os.path.join(downloadDir,checkpointName+fileExtension)
    if not os.path.exists(fileName):
        if not os.path.exists(downloadDir):
            os.makedirs(downloadDir)
        checkpointFile, _ = urllib.request.urlretrieve(url, filename=fileName,
                                                       reporthook=_print_download_progress)
        # Unpack the tar-ball
        print("Extracting Imagenet weights...")
        tarFile = tarfile.open(name=checkpointFile, mode="r:gz")
        # Get tar repo files as members.
        # We ommit the first element as it is a Directory
        tarFileList = tarFile.getmembers()[1:]
        for tar in tarFileList:
            extension =  os.path.splitext(tar.name)[1]
            if "ckpt" not in extension and extension in ckptExtension:
               extension = ".ckpt" + extension
            tar.name = checkpointName + extension 
        tarFile.extractall(downloadDir)
        tarFile.close()
        urllib.request.urlcleanup()

        print("Finished extraction")
    
    else:
        print("Imagenet weights are located in job_folder/imagenet_weights")