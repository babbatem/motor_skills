class CIP(object):
    """Composable Interaction Primitive"""
    def __init__(self):
        super(CIP, self).__init__()

    def execute_body(self):
        raise NotImplementedError

    def execute_head(self):
        raise NotImplementedError

    def sample_init_set(self):
        raise NotImplementedError

    def update_init_set(self):
        raise NotImplementedError

    def sample_effect_distribution(self):
        raise NotImplementedError

    def estimate_effect_distribution(self, samples):
        raise NotImplementedError

    def success_predicate(self):
        raise NotImplementedError

    def learning_cost(self):
        raise NotImplementedError

    def learn_body(self):
        raise NotImplementedError

    def learning_reset(self):
        raise NotImplementedError

    def get_action(self, action, policy_step):
        raise NotImplementedError
