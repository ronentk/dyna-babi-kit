from .Entity import Entity


class Location(Entity):
    def __init__(self, world, name):
        super().__init__(world, "location", name)
