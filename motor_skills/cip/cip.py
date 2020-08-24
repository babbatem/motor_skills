class CIP(object):
    """Composable Interaction Primitive"""
    def __init__(self):
        super(CIP, self).__init__()

    def body():
        raise NotImplementedError

    def head():
        raise NotImplementedError

    def init():
        raise NotImplementedError

    def effect():
        raise NotImplementedError

    def goal():
        raise NotImplementedError

    def learning_cost():
        raise NotImplementedError

    def learn_body():
        raise NotImplementedError
