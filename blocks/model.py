
class Model(object):
    """Model is a container for a set of bricks that form a single entity."""

    def __init__(self, top_bricks):
        self.top_bricks = top_bricks

    def get_params(self):
        def recursion(brick):
            result = self.params
            for child in brick.children:
                result.append(recursion(child))
            return result
        return sum([recursion(top_brick) for top_brick in self.top_bricks], [])
