from ..helpers.event import Event, BeliefType

class Entity(object):
    def __init__(self, world, kind, name=""):
        self.world = world
        self.kind = kind
        self.name = name
        self.holds = set()
        self.holder = None
        self.history = {}
        self.last_known_loc = []
        self.indef_loc_ev = []

    def reset(self):
        self.holds = set()
        self.holder = None
        self.history = {}
        self.last_known_loc = []
        self.indef_loc_ev = []
        
    def __str__(self):
        return self.name
    
    def known_events(self, reverse: bool = False,
                     start: int = None,
                      end: int = None):
        end = max(self.history.keys()) if not end else end
        start = min(self.history.keys()) if not start else start
        return sorted([(t, e) for t, e in self.history.items() if ((e.is_known) 
                                                                   and (t >= start) 
                                                                   and (t <= end))],
                      reverse=reverse
                      )
    
    @property
    def last_known_loc_support(self):
        return [e.timestep for e in self.last_known_loc]

    @property
    def indef_loc(self):
        """ 
        Return True if current location of object is not definitely known - 
        either through an indef or negate action
        """
        # either had indef event or no known location until now
        return len(self.indef_loc_ev) > 0 or len(self.last_known_loc) == 0

    def update_event(self, ev, update_last_known_loc: bool = False):
        """
        

        Parameters
        ----------
        ev : TYPE
            DESCRIPTION.
        update_last_known_loc : bool, optional
            DESCRIPTION. The default is False.
        indef_loc : bool, optional
            True if location is now indefinite as a result of this event. The default is False. Cannot be True along with update_last_known_loc.

        Returns
        -------
        None.

        """
        self.history[ev.timestep] = ev
        if ev.gold_belief != BeliefType.KNOWN:
            if ev.kind in ["indef", "negate"]:
                self.indef_loc_ev = [ev]
        if update_last_known_loc:
            self.indef_loc_ev = []
            self.last_known_loc = [ev]
            if ev.is_coref:

                # take prev event as a supporting fact if current was coref
                prev = self.world.history[ev.timestep-1]
                self.last_known_loc.insert(0, prev)
                
            # propagate info backwards to resolve unknown location of previous events involved in
            for t, prev_ev in self.known_events(end=ev.timestep-1, reverse=True):

                if prev_ev.location != ev.location:
                    break
                if prev_ev.loc_event_idxs:
                    break
                else: 
                    prev_ev.loc_event_idxs = ev.loc_event_idxs
            
            # propagate location info to other held items
            for held in self.holds:
                held.update_event(ev, update_last_known_loc=update_last_known_loc)
        
