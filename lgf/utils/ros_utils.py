import rospkg

def get_package_path(path):
    rospack = rospkg.RosPack()
    if path.startswith('package://'):
        path = path.replace('package://', '')
        path = path.split('/')
        path = rospack.get_path(path[0]) + '/' + '/'.join(path[1:])
    return path
