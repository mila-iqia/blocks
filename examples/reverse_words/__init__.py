from blocks.bricks.base import application
from blocks.bricks.recurrent import GatedRecurrent

class Transition(GatedRecurrent):
    def __init__(self, attended_dim, **kwargs):
        super(Transition, self).__init__(**kwargs)
        self.attended_dim = attended_dim

    @application(contexts=['attended', 'attended_mask'])
    def apply(self, *args, **kwargs):
        for context in Transition.apply.contexts:
            kwargs.pop(context)
        return super(Transition, self).apply(*args, **kwargs)

    @apply.delegate
    def apply_delegate(self):
        return super(Transition, self).apply

    def get_dim(self, name):
        if name == 'attended':
            return self.attended_dim
        if name == 'attended_mask':
            return 0
        return super(Transition, self).get_dim(name)


