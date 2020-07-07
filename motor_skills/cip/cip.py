class CIP(object):
    """Composable Interaction Primitive"""
    def __init__(self):
        super(CIP, self).__init__()
        
    @property
    def qpos(self):
        raise NotImplementedError

    @property
    def qvel(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError
