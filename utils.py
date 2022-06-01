import os 

def checkdir(directory):
    """ Creates file directory if it does not yet exist. 
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Make dir: %s'%directory)