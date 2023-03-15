from .Entity import Entity


class Object(Entity):
    def __init__(self, world, name):
        super().__init__(world, "object", name)
        self.last_pos_ev = []
    
    def reset(self):
        super().reset()
        self.last_pos_ev = []
        
    @property    
    def is_held(self):
        return self.holder.kind == "person"

    

    def first_known_time_at_loc(self, loc_name, time = None):
        """
        Return event where obj was first known to be at location with name `loc_name`.
        If `time` is provided and is a timestep where obj was at `loc_name`, find first
        timestep within that cluster. Otherwise assume last cluster is the relevant one.
        Example:
            For locations per timestep as follows:
            1 2 3 4 5 6
            a a b b a a
            >>> obj.first_known_time_at_loc(a, 2) # obj.history[1]
            >>> obj.first_known_time_at_loc(a) # obj.history[5], defaulting to last cluster
            
        Parameters
        ----------
        loc_name : TYPE
            DESCRIPTION.
        time : TYPE, optional
            DESCRIPTION. The default is None.



        """
        # find last cluster where obj at loc_name
        if not time or self.history[time].location != loc_name:
            for t, ev in reversed(self.known_events()):
                if ev.location == loc_name and ev.loc_knowledge_supported:
                    time = ev.timestep
                    break
        
        # else time and self.history[time].location == loc_name:
        idxs, evs = zip(*self.known_events())
        earliest = time
        for i in reversed(idxs[0:idxs.index(time)]):
            if self.history[i].location == loc_name:
                earliest = i
            else:
                # previous location, return earliest time
                return earliest
        return earliest

    def update_event(self, ev, update_last_known_loc: bool = False,
                     update_last_known_change_pos: bool = False):
        super().update_event(ev, update_last_known_loc)
        # event involves change of possession
        if update_last_known_change_pos:
            self.last_pos_ev = [ev]
            if ev.is_coref:
            #     # take prev event as a supporting fact if current was coref
                prev = self.world.history[ev.timestep-1]
                self.last_pos_ev.insert(0, prev)