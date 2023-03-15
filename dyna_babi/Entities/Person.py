from .Entity import Entity

class Person(Entity):
    def __init__(self, world, name):
        super().__init__(world, "person", name)

    def grab(self, object):
        object.holder.holds.remove(object)
        object.holder = self
        self.holds.add(object)

    def drop(self, object):
        object.holder = self.holder
        self.holder.holds.add(object)
        self.holds.remove(object)

    def move(self, location):
        self.holder.holds.remove(self)
        location.holds.add(self)
        self.holder = location

    def give(self, object, other):
        self.holds.remove(object)
        other.holds.add(object)
        object.holder = other
