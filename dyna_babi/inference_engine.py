
from enum import Enum
from collections import namedtuple
# from dataclasses import dataclass, asdict, field

# from babi_generator.Entities.Entity import Entity
# from babi_generator.Entities.Location import Location
# from babi_generator.Entities.Person import Person
# from babi_generator.Entities.Object import Object
from dyna_babi.Actions.Action import ActionType
from dyna_babi.helpers.event import Event, QuestionEvent  # , BeliefType
from dyna_babi.Questions.Question import QuestionType
from dyna_babi.world import World

class PrepositionType(int, Enum):
    AT = 0
    HOLDS = 1
    COLLOCATED = 2

    @property    
    def to_str(self):
        return prop_map.get(self)


prop_map = {
    PrepositionType.AT: "at",
    PrepositionType.HOLDS: "holds",
    PrepositionType.COLLOCATED: "collocated"
    }        


class Preposition:
    """

    """
    def __init__(self, prep_type: PrepositionType, subject: str, info: str,
                 confidence: float, supporting_facts: list):
        """
        <subject> and <info> are both names
        supporting_facts - list of set[int] (e.g., [{4}, {1,2}], or [{1}])
        """
        if prep_type == PrepositionType.AT:
            # assert isinstance(subject, (Person, Object)) and isinstance(info, Location)
            assert confidence in (0, 1)
        elif prep_type == PrepositionType.HOLDS:
            # assert isinstance(subject, Person) and isinstance(info, Object)
            assert confidence in (0, 1)
        else:   # type == PrepositionType.COLLOCATED
            # assert isinstance(subject, Person) and isinstance(info, Person)
            assert subject != info
            assert confidence == 1
        self.type_ = prep_type
        self.subject = subject
        self.info = info
        self.confidence = confidence
        self.s_facts = supporting_facts     # list of sets
        self.clean_s_facts()

    def similar(self, x):
        """
        ~equality operator. ignores <s_facts>
        """
        return self.type_ == x.type_ and self.subject == x.subject and \
            self.info == x.info and self.confidence == x.confidence

    def copy(self):
        res = Preposition(self.type_, self.subject, self.info,
                          self.confidence, self.s_facts.copy())
        return res

    def switch_subject(self, subj):
        """
        constructs a copy of <self> with an updated subject
        """
        res = self.copy()
        res.subject = subj
        return res
    
    def __str__(self):
        return f"{self.type_.to_str}({self.subject},{self.info}), c={self.confidence}"
    
    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def combine_lists(l1: list, l2: list):
        """
        Combines two lists of sets, by uniting all combinations from both lists.
        E.g. for <l1>==[{3}, {1,2}] and <l2>==[{4}, {1,2}]
        returns: [{3,4}, {1,2,3}, {1,2,4}, {1,2}]
        """
        return [set1.union(set2) for set1 in l1 for set2 in l2]

    def append_all_facts(self, new_facts: list):
        """
        Combines a new list of s_facts with this Preposition <s_facts>
        (see: Preposition.combine_lists)
        new_facts - list of set[int] (e.g., [{4}, {1,2}])
        """
        self.s_facts = Preposition.combine_lists(self.s_facts, new_facts)
        self.clean_s_facts()

    def add_new_s_facts(self, new_facts: list):
        """
        Concatenates a new list of s_facts to this Preposition <s_facts>
        new_facts - list of set[int] (e.g., [{4}, {1,2}])
        """
        self.s_facts += new_facts
        self.clean_s_facts()

    def clean_s_facts(self):
        """"
        Makes sure <s_facts> does not contain any duplications or supersets
        """
        Preposition.remove_supersets(self.s_facts)

    @staticmethod
    def remove_supersets(s_facts):
        """"
        An in-place auxilary function.
        Makes sure a list of sets does not contain any duplications or supersets
        """
        redundant_inds = []
        for ind in range(len(s_facts)):
            sf = s_facts[ind]
            for ind2 in range(ind):
                sf2 = s_facts[ind2]
                if sf <= sf2:
                    redundant_inds.append(ind2)
                elif sf2 < sf:
                    redundant_inds.append(ind)
                    break  # <sf> will be removed anyway
        # remove duplications and supersets
        redundant_inds = list(set(redundant_inds))  # remove indices duplications
        for ind in reversed(sorted(redundant_inds)):
            # erasing in reversed order, so we don't change the list
            s_facts.pop(ind)


class InferenceEngine:
    """
        events_history -
        prepositions_history -
    """
    def __init__(self):
        self.events_history = []
        self.prepositions_history = []

    def reset(self):
        self.events_history = []
        self.prepositions_history = []

    def add_event(self, event: Event):
        """
        Adds a given <event> to events history and upadte the prepositions
        history accordingly
        """
        self.events_history.append(event)
        # 1. get prepositions from current event
        event_prepositions = self.infer_from_event(event)

        # 2. merge the event's prepositions into existing prepositions
        prepositions_copy = []
        if self.prepositions_history:  # copy the last frame's prepositions
            prepositions_copy = [p.copy() for p in self.prepositions_history[-1]]
        self.prepositions_history.append(prepositions_copy)
        for new_prep in event_prepositions:
            self.merge_new_prepositions(new_prep)

        # 3. infer locations from collocation prepositions
        self.solve_collocations()

        # 4. maintenance
        for prep in self.prepositions_history[-1]:
            prep.clean_s_facts()             # remove unneeded supporting facts
        self.check_validity()

    def add_all_events(self, world: World):
        """
        Adds all the events in the history of a given <world>
        """
        inds = sorted(world.history.keys())
        for ind in inds:
            self.add_event(world.history[ind])

    @staticmethod
    def desc_event(ev: Event):
        """
        Returns a string description of a given Event
        TODO: move to Event class?
        """
        if ev.kind in (ActionType.MOVE, ActionType.GRAB, ActionType.DROP,
                       ActionType.INDEF, ActionType.GIVE_ALL):
            ev_desc = "%s(%s, %s)" % (ev.kind, ev.source, ev.target)
        elif ev.kind == ActionType.NEGATE:
            # gold = "NEG" if ev.gold_belief == BeliefType.NEGATED else "MOVE"
            gold = "NEG" if ev.target != ev.location else "MOVE"
            ev_desc = "%s(%s, %s) [%s]" % (ev.kind, ev.source, ev.target, gold)
        elif ev.kind == ActionType.GIVE:
            ev_desc = "%s(%s, %s, %s)" % (ev.kind, ev.source, ev.target, ev.ternary)
        elif ev.kind == ActionType.DROP_ALL:
            ev_desc = "%s(%s)" % (ev.kind, ev.source)
        else:           # ev.kind in (COREF, CONJ, COMPOUND)
            assert False

        coref_str = ""
        if ev.is_coref:
            coref_str = "[coref]"
        return "%d %s %s" % (ev.timestep, ev_desc, coref_str)

    @staticmethod
    def desc_question(question: QuestionEvent):
        """
        Returns a string description of a given QuestionEvent
        TODO: move to QuestionEvent class?
        """
        if question.kind == QuestionType.YES_NO:
            q_desc = "%s %s in %s" % (question.kind, question.source, question.ternary)
        elif question.kind == QuestionType.WHERE_WAS_OBJ:
            q_desc = "%s %s before %s" % (question.kind, question.source, question.ternary)
        elif question.kind == QuestionType.GIVING:
            if question.sub_kind == "gave_to":
                q_desc = "Who gave the %s to %s" % (question.source, question.ternary)
            elif question.sub_kind == "gave":
                q_desc = "Who gave the %s" % question.source
            elif question.sub_kind == "received":
                q_desc = "Who received the %s" % question.source
            elif question.sub_kind == "who_give":
                q_desc = "Who did %s give the %s to" % (question.source, question.ternary)
            else:
                assert question.sub_kind == "what_give"
                q_desc = "What did %s give to %s" % (question.source, question.ternary)
        else:
            q_desc = "%s %s" % (question.kind, question.source)
        return "%d QUESTION: %s?" % (question.timestep, q_desc)

    @staticmethod
    def convert_to_text(num):
        """
        a small auxilary function to translate integers to
        """
        assert isinstance(num, int) and 0 <= num <= 10
        if num == 10:
            return 'ten'
        if num == 9:
            return 'nine'
        if num == 8:
            return 'eight'
        if num == 7:
            return 'seven'
        if num == 6:
            return 'six'
        if num == 5:
            return 'five'
        if num == 4:
            return 'four'
        if num == 3:
            return 'three'
        if num == 2:
            return 'two'
        if num == 1:
            return 'one'
        return 'none'

    @staticmethod
    def solve_where_question(entity, prepositions):
        """
        Returns the location [list of strings] of a given <entity> according to
        given <prepositions>
        """
        locs = InferenceEngine.get_location_info(prepositions, entity, confidence=1)

        # case I: we have certain location of <entity>
        sure_preps = [prep for prep in locs if len(prep.info) == 1]
        if sure_preps:
            assert len(sure_preps) == 1
            return sure_preps[0].info, sure_preps[0].s_facts

        # case II: we only have uncertain location of <entity> (after Indef)
        maybe_preps = [prep for prep in locs if len(prep.info) > 1]
        if maybe_preps:
            assert len(maybe_preps) == 1
            return maybe_preps[0].info, maybe_preps[0].s_facts

        # TODO: return "not in [L_1,...L_n]" if such prepositions exist
        return ["UNKNOWN"], []

    def solve_yesno_question(self, entity, location):
        """
        Solves a yes/no question - whether <entity> is in a specific <location>

        """
        curr_preps = self.prepositions_history[-1]
        at_preps = InferenceEngine.get_location_info(curr_preps, entity)

        # 1. extract all answers from all of <entity>'s AT preps
        answers, sfs = [], []
        for at_prep in at_preps:
            # 1.1. exclusion AT prep - subject is not at <info>
            if at_prep.confidence == 0:
                if at_prep.info[0] == location:   # otherwise - ignore
                    answers.append("no")
                    sfs.append(at_prep.s_facts)
            # 1.2. preps of known location
            elif len(at_prep.info) == 1:
                ans = "yes" if at_prep.info[0] == location else "no"
                answers.append(ans)
                sfs.append(at_prep.s_facts)
            else:  # 1.3. maybe location; len(at_prep.info) > 1
                ans = "maybe" if location in at_prep.info else "no"
                answers.append(ans)
                sfs.append(at_prep.s_facts)

        # 2. collect answers to groups
        y_ind = [ind for ind, ans in enumerate(answers) if ans == "yes"]
        n_ind = [ind for ind, ans in enumerate(answers) if ans == "no"]
        m_ind = [ind for ind, ans in enumerate(answers) if ans == "maybe"]
        assert not n_ind or not y_ind       # can't co-exist
        assert n_ind or y_ind or m_ind        # current preps should give an answer

        # 3.1. "yes" answer exists
        if y_ind:
            assert len(y_ind) == 1
            return "yes", sfs[y_ind[0]]

        # 3.2. "no" answer exists
        if n_ind:
            sfs = [sfs[ind] for ind in n_ind]
            sfs_concat = [sf for sf_list in sfs for sf in sf_list]
            return "no", sfs_concat

        # 3.3. "maybe"
        sfs_concat = [sf for sf_list in sfs for sf in sf_list]
        return "maybe", sfs_concat

    def solve_giving_question(self, source, ternary, subkind):
        """
        Solves a GivingQuestion event based on its source and ternary
        The possible values of <subkind> determine the exact question:
        - "gave": "Who gave the %s?" % question.source
        - "received": "Who received the %s?" % question.source
        - "gave_to": "Who gave the %s to %s?" % (question.source, question.ternary)
        - "who_give": "Who did %s give the %s to?" % (question.source, question.ternary)
        - "what_give": "What did %s give to %s?" % (question.source, question.ternary)

        * Code can be written more efficiently
        """

        assert subkind in ("gave", "received", "gave_to", "who_give", "what_give")
        Holding = namedtuple("Holding", "holder obj sf")

        # initializations
        preps = self.prepositions_history
        prev_holdings = [Holding(prep.subject, prep.info, prep.s_facts)
                         for prep in InferenceEngine.get_possesion_info(preps[-1],
                                                                        confidence=1)]
        # loop over history to look for giving event
        for ind in range(len(self.prepositions_history)-1, 0, -1):
            curr_holdings = prev_holdings
            prev_holdings = [Holding(prep.subject, prep.info, prep.s_facts)
                             for prep in InferenceEngine.get_possesion_info(preps[ind-1],
                                                                            confidence=1)]

            # find all giving events in current time-step
            swaps = [(prev_h, curr_h)
                     for prev_h in prev_holdings for curr_h in curr_holdings
                     if prev_h.obj == curr_h.obj and prev_h.holder != curr_h.holder]
            if not swaps:           # no giving event on time-step
                continue
            assert len(swaps) == 1      # only one giving event can occur

            #
            giver = swaps[0][0].holder
            reciver = swaps[0][1].holder
            obj = swaps[0][0].obj
            sf = swaps[0][1].sf

            # "gave": "Who gave the %s?" % source
            # "received": "Who received the %s?" % source
            if subkind in ("gave", "received") and source == obj:
                return giver if subkind == "gave" else reciver, sf

            # "gave_to": "Who gave the %s to %s?" % (source, ternary)
            if subkind == "gave_to" and source == obj and ternary == reciver:
                return giver, sf

            # "who_give": "Who did %s give the %s to?" % (source, ternary)
            if subkind == "who_give" and source == giver and ternary == obj:
                return reciver, sf

            # "what_give": "What did %s give to %s?" % (source, ternary)
            if subkind == "what_give" and source == giver and ternary == reciver:
                return obj, sf

        assert False  # relevant GiveAction was not found
        # return None, []

    def solve_question(self, question: QuestionEvent):
        """
        Solves a given QuestionEvent using the prepositions history
        Returns: ans [str]
                 s_facts [list of sets] - the possible supporting facts needed to
                    solve <question>
        """
        # initialization
        curr_preps = self.prepositions_history[-1]
        s_facts = [set()]
        ans = ""

        # 1. CountingQuestion - How many objects is {} carrying?
        #    ListQuestion - What is {} carrying?
        if question.kind in (QuestionType.COUNT, QuestionType.LIST):
            subj_poss = InferenceEngine.get_possesion_info(curr_preps,
                                                           question.source, confidence=1)
            subj_all_poss = InferenceEngine.get_possesion_info(curr_preps,
                                                               question.source)
            if question.kind == QuestionType.COUNT:
                ans = InferenceEngine.convert_to_text(len(subj_poss))
            else:
                ans = [sorted([prep.info for prep in subj_poss])]
            # collect supporting facts
            for prep in subj_all_poss:
                assert len(prep.s_facts) == 1
                s_facts = Preposition.combine_lists(s_facts, prep.s_facts)
        # 2a. WhereObjectQuestion - Where is the {}?
        # 2b. WherePersonQuestion - Where is {}?
        elif question.kind in (QuestionType.WHERE_OBJ, QuestionType.WHERE_PERSON):
            ans, s_facts = InferenceEngine.solve_where_question(question.source,
                                                                curr_preps)
        # 2c. YesNoQuestion - Is {} in the {}?
        elif question.kind == QuestionType.YES_NO:
            assert isinstance(question.ternary, str)  # and not a tuple of locations
            ans, s_facts = self.solve_yesno_question(question.source, question.ternary)

        # 3. GivingQuestion - see <solve_giving_question>
        elif question.kind == QuestionType.GIVING:
            ans, s_facts = self.solve_giving_question(question.source, question.ternary,
                                                      question.sub_kind)
        # 4. WhereWasObjectQuestion - Where was the {} before the {}?
        elif question.kind == QuestionType.WHERE_WAS_OBJ:
            visited_loc = False
            loc_s_facts = []
            for preps in reversed(self.prepositions_history):
                res, res_s_facts = InferenceEngine.solve_where_question(question.source,
                                                                        preps)
                in_loc = len(res) == 1 and res[0] == question.ternary
                if visited_loc and not in_loc:
                    # found the object's location before the specified one
                    assert res[0] != "UNKNOWN"
                    ans = res
                    s_facts = Preposition.combine_lists(res_s_facts, loc_s_facts)
                    break
                if in_loc:
                    # mark that the object is in the specified location
                    visited_loc = True
                    loc_s_facts = res_s_facts
            assert ans != ""     # previous position was found
        else:
            assert False

        # output
        if not isinstance(ans, list):
            ans = [ans]
        Preposition.remove_supersets(s_facts)
        return ans, s_facts

    def infer_from_event(self, ev: Event):
        """
        Calculates a list of Prepositions derived from a given new event.
        Returns: list of Prepositions
        """
        # 0. initializations
        curr_preps = self.prepositions_history[-1] if self.prepositions_history else []
        res = []
        assert not ev.is_q         # questions are handled in <solve_question>

        # 1. set supporting facts for this event (see in Preposition)
        #    usually it's the current statement. in coreference it's also the prev one
        sf = [{ev.timestep - 1, ev.timestep}] if ev.is_coref else [{ev.timestep}]
        assert ev.timestep > 1 or not ev.is_coref   # 1st sentence can't be a coreference

        # 2. return the inffered prepositions list for every type of event
        # 2.1. moving actions - add AT prepositions for all subjects
        if ev.kind in (ActionType.MOVE, ActionType.INDEF, ActionType.NEGATE):
            # assert ev.gold_belief in (BeliefType.NEGATED, BeliefType.KNOWN)
            target = ev.target if isinstance(ev.target, list) else [ev.target]
            # assert (ev.target != ev.location) == (ev.gold_belief == BeliefType.NEGATED)
            # if ev.kind == ActionType.NEGATE and ev.gold_belief == BeliefType.NEGATED:
            if ev.kind == ActionType.NEGATE and ev.target != ev.location:
                conf = 0
                subjects = [ev.source]
            else:
                conf = 1
                if ev.is_conj:
                    # when Indef-ConjAction will be supported we need to add a coloc-prep
                    assert ev.kind != ActionType.INDEF
                    assert len(ev.source) == 2   # just 2 people in conj
                    subjects = ev.source
                else:
                    subjects = [ev.source]
            # subject(s) AT prepositions
            for subj in subjects:
                res.append(Preposition(PrepositionType.AT, subj, target,
                                       confidence=conf, supporting_facts=sf))
        # 2.2. grab action - add HOLDS and holder-object collocation
        elif ev.kind == ActionType.GRAB:
            res.append(Preposition(PrepositionType.HOLDS, ev.source, ev.target,
                                   confidence=1, supporting_facts=sf))
            res.append(Preposition(PrepositionType.COLLOCATED, ev.source, ev.target,
                                   confidence=1, supporting_facts=sf))
        # 2.3. drop actions - add HOLDS (conf==0) and holder-object collocation
        elif ev.kind in (ActionType.DROP, ActionType.DROP_ALL):
            if ev.kind == ActionType.DROP:
                # <DROP_ALL> is sometimes implemented as DROP with a list of objects
                objs_dropped = ev.target if isinstance(ev.target, list) else [ev.target]
            else:
                all_poss = InferenceEngine.get_possesion_info(curr_preps,
                                                              ev.source, confidence=1)
                objs_dropped = [prep.info for prep in all_poss]
            # add prepositions
            for obj in objs_dropped:
                res.append(Preposition(PrepositionType.HOLDS, ev.source, obj,
                                       confidence=0, supporting_facts=sf))
                res.append(Preposition(PrepositionType.COLLOCATED, ev.source, obj,
                                       confidence=1, supporting_facts=sf))
        # 2.4. give actions - update two HOLDS prepositions (grabbing and dropping) and
        #      two person-object collocations
        elif ev.kind in (ActionType.GIVE, ActionType.GIVE_ALL):
            if ev.kind == ActionType.GIVE:
                # <GIVE_ALL> is sometimes implemented as GIVE with a list of objects
                objs_given = ev.ternary if isinstance(ev.ternary, list) else [ev.ternary]
            else:
                all_poss = InferenceEngine.get_possesion_info(curr_preps,
                                                              ev.source, confidence=1)
                objs_given = [prep.info for prep in all_poss]
            # add prepositions
            for obj in objs_given:
                res.append(Preposition(PrepositionType.HOLDS, ev.source, obj,
                                       confidence=0, supporting_facts=sf))
                res.append(Preposition(PrepositionType.COLLOCATED, ev.source, obj,
                                       confidence=1, supporting_facts=sf))
                res.append(Preposition(PrepositionType.HOLDS, ev.target, obj,
                                       confidence=1, supporting_facts=sf))
                res.append(Preposition(PrepositionType.COLLOCATED, ev.target, obj,
                                       confidence=1, supporting_facts=sf))
        #
        assert ev.kind not in (ActionType.COREF, ActionType.CONJ, ActionType.COMPOUND)
        return res

    def merge_at_preposition(self, at_prep: Preposition):
        """
        merges a new AT preposition into the latest prepositions list.
        """
        curr_prepositions = self.prepositions_history[-1]
        known_locs = InferenceEngine.get_location_info(curr_prepositions,
                                                       at_prep.subject)
        subj_poss = InferenceEngine.get_possesion_info(curr_prepositions,
                                                       subject=at_prep.subject,
                                                       confidence=1)
        # 1. replace old location info with new preposition
        if known_locs:
            assert len(known_locs) != 1 or not known_locs[0].similar(at_prep)
            for p in known_locs:              # remove known location info
                curr_prepositions.remove(p)
        curr_prepositions.append(at_prep)

        # 2. break the subject's collocation
        subj_collocation = \
            InferenceEngine.get_coloc_info(curr_prepositions, at_prep.subject)
        for coloc in subj_collocation:
            # don't break subject's collocation with held objects
            if not any([prep.info == coloc.info for prep in subj_poss]):
                curr_prepositions.remove(coloc)

        # 3. update info for held objects
        for p_poss in subj_poss:
            # obj = prep.info
            obj_locs = InferenceEngine.get_location_info(curr_prepositions, p_poss.info)
            obj_colocs = InferenceEngine.get_coloc_info(curr_prepositions, p_poss.info)

            # 3.1. update their location
            for p_loc in obj_locs:  # remove known location info
                curr_prepositions.remove(p_loc)
            # add updated location info
            sf = Preposition.combine_lists(p_poss.s_facts, at_prep.s_facts)
            new_obj_loc = Preposition(PrepositionType.AT, p_poss.info, at_prep.info,
                                      confidence=at_prep.confidence, supporting_facts=sf)
            curr_prepositions.append(new_obj_loc)

            # 3.2. remove their collocations
            for p_coloc in obj_colocs:
                if p_coloc.subject != at_prep.subject:  # don't break coloc with holder
                    curr_prepositions.remove(p_coloc)

            # 3.3. if new position is not certain - add collocation to holder
            if at_prep.confidence != 1 or len(at_prep.info) > 1:
                if all([p.subject != at_prep.subject for p in obj_colocs]):
                    obj_coloc = Preposition(PrepositionType.COLLOCATED,
                                            at_prep.subject, p_poss.info, confidence=1,
                                            supporting_facts=p_poss.s_facts)
                    curr_prepositions.append(obj_coloc)

        # 4. add s_fact to AT preposition of dropped objects
        drop_preps = InferenceEngine.get_possesion_info(curr_prepositions,
                                                        subject=at_prep.subject,
                                                        confidence=0)
        for p_drop in drop_preps:
            if any([prep.info == p_drop.info for prep in subj_collocation]):
                # i.e., subject is leaving the place where the object was dropped
                obj_locs = InferenceEngine.get_location_info(curr_prepositions,
                                                             p_drop.info)
                # add the subject_dropping to object location s_facts
                for p_loc in obj_locs:
                    p_loc.append_all_facts(p_drop.s_facts)

    def merge_holds_preposition(self, new_prep: Preposition):
        """
        merges a new HOLDS preposition into the latest prepositions list.
        """
        curr_prepositions = self.prepositions_history[-1]

        # update possession info
        obj = new_prep.info
        obj_poss = InferenceEngine.get_possesion_info(curr_prepositions,
                                                      obj=obj, confidence=1)
        if not obj_poss:
            # case 1: no one is holding the object
            assert new_prep.confidence == 1  # must be picked
            # in case the same subject dropped the object before
            outdated_poss = InferenceEngine.get_possesion_info(curr_prepositions,
                                                               subject=new_prep.subject,
                                                               obj=obj)
            if outdated_poss:
                assert len(outdated_poss) == 1 and outdated_poss[0].confidence == 0
                curr_prepositions.remove(outdated_poss[0])
            #
            curr_prepositions.append(new_prep)
        else:  # compare known info and new info
            # case 2: someone is holding the object
            assert new_prep.confidence == 0  # must be dropped (DROP / part of GIVE)
            assert len(obj_poss) == 1
            obj_poss = obj_poss[0]  # bad notation here

            # HOLDS preposition for dropped objs are used also for Count/List questions
            if obj_poss.similar(new_prep):   # we already know this object was dropped
                obj_poss.add_new_s_facts(new_prep.s_facts)
            else:  # other person dropped this object earlier,
                curr_prepositions.remove(obj_poss)
                curr_prepositions.append(new_prep)

    def merge_coloc_preposition(self, new_prep: Preposition):
        """
        merges a new COLLOCATION preposition into the latest prepositions list.
        """
        curr_prepositions = self.prepositions_history[-1]

        # as collocations are solved in <solve_collocations>, just add
        # the new collocations, and update s_facts if the coloc already exists
        known_coloc = InferenceEngine.get_coloc_info(curr_prepositions,
                                                     new_prep.subject,
                                                     new_prep.info)
        if not known_coloc:
            curr_prepositions.append(new_prep)
        else:
            assert len(known_coloc) == 1
            known_coloc[0].add_new_s_facts(new_prep.s_facts)

    def merge_new_prepositions(self, new_prep: Preposition):
        """
        merges a new  preposition into the latest prepositions list.
        """
        if new_prep.type_ == PrepositionType.AT:
            # 1. move/indef/negate actions
            self.merge_at_preposition(new_prep)
        elif new_prep.type_ == PrepositionType.HOLDS:
            # 2. grab/drop/drop_all/give/give_all actions
            self.merge_holds_preposition(new_prep)
        else:
            assert new_prep.type_ == PrepositionType.COLLOCATED
            # 3. collocation from a give/give_all action
            self.merge_coloc_preposition(new_prep)

    @staticmethod
    def union_all_negates(p1, p2, negate_preps1, negate_preps2, s_facts):
        """
        Joins two lists of Prepositions created from NegateActions for 2 people
        p1, p2 -
        negate_preps1, negate_preps2 -
        s_facts -
        """
        # initializaions
        res = []

        # add preps from <p1_negate_loc>, including intersections
        for prep in negate_preps1:
            prep2 = [p for p in negate_preps2 if p.info == prep.info]
            if not prep2:
                res.append(Preposition(PrepositionType.AT, p1, prep.info, 0,
                                       prep.s_facts))
                res.append(Preposition(PrepositionType.AT, p2, prep.info, 0,
                                       Preposition.combine_lists(prep.s_facts,
                                                                 s_facts)))
            else:
                prep2 = prep2[0]        # bad notation
                res.append(Preposition(PrepositionType.AT, p1, prep.info, 0,
                                       prep.s_facts +
                                       Preposition.combine_lists(prep2.s_facts,
                                                                 s_facts)))
                res.append(Preposition(PrepositionType.AT, p2, prep.info, 0,
                                       prep2.s_facts +
                                       Preposition.combine_lists(prep.s_facts,
                                                                 s_facts)))

        # add preps from <p1_negate_loc>, including intersections
        for prep in negate_preps2:
            if not any([p for p in negate_preps1 if p.info == prep.info]):
                res.append(Preposition(PrepositionType.AT, p1, prep.info, 0,
                                       Preposition.combine_lists(prep.s_facts,
                                                                 s_facts)))
                res.append(Preposition(PrepositionType.AT, p2, prep.info, 0,
                                       prep.s_facts))

        return res

    @staticmethod
    def same_locations(p1_at: list, p2_at: list):
        """
        Comapres the location info of two entities and returns True iff it is equal
        p1_at, p2_at: lists of AT prepositions for comparison
        """

        # empty case
        if not p1_at and not p2_at:
            return True
        conf1_preps1 = [prep for prep in p1_at if prep.confidence == 1]
        conf1_preps2 = [prep for prep in p2_at if prep.confidence == 1]

        # compare certain location
        sure_preps1 = [prep for prep in conf1_preps1 if len(prep.info) == 1]
        sure_preps2 = [prep for prep in conf1_preps2 if len(prep.info) == 1]
        assert len(sure_preps1) <= 1 and len(sure_preps2) <= 1
        if sure_preps1 and sure_preps2:
            return sure_preps1[0].info == sure_preps2[0].info
        if sure_preps1 or sure_preps2:    # one is empty and the other is not
            return False

        # compare indef locations
        maybe_preps1 = [prep for prep in conf1_preps1 if len(prep.info) > 1]
        maybe_preps2 = [prep for prep in conf1_preps2 if len(prep.info) > 1]
        assert len(maybe_preps1) <= 1 and len(maybe_preps2) <= 1
        if maybe_preps1 and maybe_preps2:
            return set(maybe_preps1[0].info) == set(maybe_preps2[0].info)
        if maybe_preps1 or maybe_preps2:   # one is empty and the other is not
            return False

        # compare non-locations
        exclude_preps1 = [prep for prep in p1_at if prep.confidence == 0]
        exclude_preps2 = [prep for prep in p2_at if prep.confidence == 0]
        if exclude_preps1 and exclude_preps2:
            return set([prep.info[0] for prep in exclude_preps1]) == \
                   set([prep.info[0] for prep in exclude_preps2])

        assert exclude_preps1 or exclude_preps2    # we check that both list aren't empty
        return False

    @staticmethod
    def merge_locations(prepositions: list, p1: str, p2: str, s_facts: list):
        """
        Merges locations knowledge from given prepositions of two given entities
        prepositions -
        p1, p2: two entities who we wish to infer their collocation
        s_facts: list of sets. see in Preposition documentation
        Returns a list
        """

        # initializations
        res = []

        conf1_preps1 = InferenceEngine.get_location_info(prepositions, p1, 1)
        conf1_preps2 = InferenceEngine.get_location_info(prepositions, p2, 1)
        sure_preps1 = [prep for prep in conf1_preps1 if len(prep.info) == 1]
        sure_preps2 = [prep for prep in conf1_preps2 if len(prep.info) == 1]
        maybe_preps1 = [prep for prep in conf1_preps1 if len(prep.info) > 1]
        maybe_preps2 = [prep for prep in conf1_preps2 if len(prep.info) > 1]
        assert len(sure_preps1) <= 1 and len(sure_preps2) <= 1
        assert len(maybe_preps1) <= 1 and len(maybe_preps2) <= 1
        exclude_preps1 = InferenceEngine.get_location_info(prepositions, p1, 0)
        exclude_preps2 = InferenceEngine.get_location_info(prepositions, p2, 0)
        sure_locs = set([prep.info[0] for prep in sure_preps1 + sure_preps2])
        exclude_locs = set([prep.info[0] for prep in exclude_preps1 + exclude_preps2])
        assert not sure_locs.intersection(exclude_locs)

        # 1. case I: known location in the original data
        if sure_preps1 or sure_preps2:

            if sure_preps1 and sure_preps2:
                p1_sf = sure_preps1[0].s_facts
                p2_sf = sure_preps2[0].s_facts
                assert sure_preps1[0].info == sure_preps2[0].info
                res.append(Preposition(PrepositionType.AT, p1, sure_preps1[0].info, 1,
                                       p1_sf + Preposition.combine_lists(p2_sf, s_facts)))
                res.append(Preposition(PrepositionType.AT, p2, sure_preps1[0].info, 1,
                                       p2_sf + Preposition.combine_lists(p1_sf, s_facts)))
            elif sure_preps1:
                p1_sf = sure_preps1[0].s_facts
                res.append(Preposition(PrepositionType.AT, p1, sure_preps1[0].info, 1, p1_sf))
                res.append(Preposition(PrepositionType.AT, p2, sure_preps1[0].info, 1,
                                       Preposition.combine_lists(p1_sf, s_facts)))
            else:
                p2_sf = sure_preps2[0].s_facts
                res.append(Preposition(PrepositionType.AT, p1, sure_preps2[0].info, 1,
                                       Preposition.combine_lists(p2_sf, s_facts)))
                res.append(Preposition(PrepositionType.AT, p2, sure_preps2[0].info, 1, p2_sf))

            return res

        # 2. case II: combining IndefAction and Neg information
        if maybe_preps1 and maybe_preps2:
            maybe_locs1 = set(maybe_preps1[0].info)
            maybe_locs2 = set(maybe_preps2[0].info)

            if maybe_locs1 == maybe_locs2:
                # case 2.1: same IndefAction locations
                copied1 = maybe_preps1[0].copy()
                copied1.add_new_s_facts(
                    Preposition.combine_lists(maybe_preps2[0].s_facts, s_facts))
                copied2 = maybe_preps2[0].copy()
                copied2.add_new_s_facts(
                    Preposition.combine_lists(maybe_preps1[0].s_facts, s_facts))
                return [copied1, copied2]

            if maybe_locs1.intersection(maybe_locs2):
                assert len(maybe_locs1.intersection(maybe_locs2)) == 1
                # case 2.2: combine "maybe" from both sources yielded a certain location
                sure_loc = maybe_locs1.intersection(maybe_locs2).pop()
                sf = Preposition.combine_lists(maybe_preps1[0].s_facts,
                                               maybe_preps2[0].s_facts)
                sf = Preposition.combine_lists(sf, s_facts)

                res.append(Preposition(PrepositionType.AT, p1, [sure_loc], 1, sf))
                res.append(Preposition(PrepositionType.AT, p2, [sure_loc], 1, sf))
                return res

        # case 2.3: intersection between indef and negate
        if maybe_preps1 or maybe_preps2:
            assert len(maybe_preps1 + maybe_preps2) == 1
            maybe_prep = (maybe_preps1 + maybe_preps2)[0]
            maybe_locs = set(maybe_prep.info)
            if maybe_locs.intersection(exclude_locs):
                assert len(maybe_locs.intersection(exclude_locs)) == 1
                exclude_prep = [prep for prep in exclude_preps1 + exclude_preps2
                                if prep.info[0] in maybe_locs][0]
                sure_loc = [loc for loc in maybe_prep.info
                            if loc != exclude_prep.info[0]][0]
                sf = Preposition.combine_lists(maybe_prep.s_facts,
                                               exclude_prep.s_facts)
                sf = Preposition.combine_lists(sf, s_facts)

                res.append(Preposition(PrepositionType.AT, p1, [sure_loc], 1, sf))
                res.append(Preposition(PrepositionType.AT, p2, [sure_loc], 1, sf))
                return res

        # case 3. copy info from IndefAction (if exists) and add all negations
        # one of these 2 loops will be empty
        for prep in maybe_preps1:
            res.append(Preposition(PrepositionType.AT, p1, prep.info, 1,
                                   prep.s_facts))
            res.append(Preposition(PrepositionType.AT, p2, prep.info, 1,
                                   Preposition.combine_lists(prep.s_facts,
                                                             s_facts)))
        for prep in maybe_preps2:
            res.append(Preposition(PrepositionType.AT, p2, prep.info, 1,
                                   prep.s_facts))
            res.append(Preposition(PrepositionType.AT, p1, prep.info, 1,
                                   Preposition.combine_lists(prep.s_facts,
                                                             s_facts)))

        res += InferenceEngine.union_all_negates(p1, p2, exclude_preps1,
                                                 exclude_preps2, s_facts)
        return res

    def solve_collocations(self):
        """
        Makes sure the current info in self.prepositions_history is coalesced:
        - infer location using collocation preprositions
        - remove undeeded collocation preprositions
        """
        # 1. update known location using collocation
        prepositions = self.prepositions_history[-1]
        coloc_preps = InferenceEngine.get_coloc_info(prepositions)
        while True:
            new_info = False  # exit condition. True iff we gain new info in curr iter
            for coloc_prep in coloc_preps:
                preps1 = InferenceEngine.get_location_info(prepositions,
                                                           coloc_prep.subject)
                preps2 = InferenceEngine.get_location_info(prepositions,
                                                           coloc_prep.info)
                #
                if not preps1 and not preps2:
                    continue
                if not preps1 or not preps2:
                    new_info = True
                    not_empty = preps1 if preps1 else preps2
                    other = coloc_prep.info if preps1 else coloc_prep.subject
                    for prep in not_empty:
                        new_prep = prep.switch_subject(other)
                        new_prep.append_all_facts(coloc_prep.s_facts)
                        prepositions.append(new_prep)
                else:
                    merged_preps = \
                        InferenceEngine.merge_locations(prepositions,
                                                        coloc_prep.subject,
                                                        coloc_prep.info,
                                                        coloc_prep.s_facts)
                    # add <merged_preps> data to <prepositions>
                    for merged_p in merged_preps:
                        existing = [old_p for old_p in (preps1 + preps2) if
                                    merged_p.similar(old_p)]
                        assert len(existing) <= 1
                        if not existing:
                            prepositions.append(merged_p)
                        else:
                            existing[0].s_facts = merged_p.s_facts

                    # update <new_info>
                    updated1 = InferenceEngine.get_location_info(merged_preps,
                                                                 coloc_prep.subject)
                    updated2 = InferenceEngine.get_location_info(merged_preps,
                                                                 coloc_prep.info)
                    new_info |= not InferenceEngine.same_locations(updated1, preps1)
                    new_info |= not InferenceEngine.same_locations(updated2, preps2)

            if not new_info:       # no point continuing this loop
                break

    @staticmethod
    def get_location_info(prepositions, subject, confidence=None):
        """
        Returns all the AT prepositions for a given subject.
        prepositions -
        subject -
        confidence -
        """
        loc_preps = [prep for prep in prepositions if
                     prep.type_ == PrepositionType.AT and prep.subject == subject]
        if confidence is None:
            return loc_preps
        assert confidence in (0, 1)
        return [prep for prep in loc_preps if prep.confidence == confidence]

    @staticmethod
    def get_possesion_info(prepositions, subject=None, obj=None, confidence=None):
        """
        Returns all the HOLDS prepositions for a given subject.
        prepositions -
        subject -
        obj -
        confidence -
        """
        all_poss = [prep for prep in prepositions if prep.type_ == PrepositionType.HOLDS]
        if subject is not None:
            all_poss = [prep for prep in all_poss if prep.subject == subject]
        if obj is not None:
            all_poss = [prep for prep in all_poss if prep.info == obj]
        if confidence is not None:
            assert confidence in (0, 1)
            all_poss = [prep for prep in all_poss if prep.confidence == confidence]
        return all_poss

    @staticmethod
    def get_coloc_info(prepositions, entity1=None, entity2=None):
        """
        Returns all the COLLOCATED prepositions for a given entity or between
        two given entities.
        prepositions - list of Prepositions to scan
        If only one of <entity1>, <entity2> is not None - returns all the prepositions
        containing that entity.
        If both <entity1>, <entity2> are None - returns all the collocation prepositions
        """
        res = []
        for prep in prepositions:
            if prep.type_ != PrepositionType.COLLOCATED:
                continue
            assert prep.confidence == 1
            assert prep.subject != prep.info     # we don't expect reflexive preps
            if entity1 is not None and entity1 not in (prep.subject, prep.info):
                continue
            if entity2 is not None and entity2 not in (prep.subject, prep.info):
                continue
            res.append(prep)

        return res

    def check_validity(self):
        """
        Verifies the prepositions in the latest entry of <prepositions_history>
        are consistent and valid.
        """
        prepositions = self.prepositions_history[-1]

        holds_preps = InferenceEngine.get_possesion_info(prepositions, confidence=1)
        held_objects = [prep.info for prep in holds_preps]
        all_entities = set([prep.subject for prep in prepositions])
        for ent in all_entities:
            conf1_preps = InferenceEngine.get_location_info(prepositions, ent, 1)
            # empty intersections
            sure_preps = [prep for prep in conf1_preps if len(prep.info) == 1]
            assert len(sure_preps) <= 1  # (at most) one approved location
            maybe_preps = [prep for prep in conf1_preps if len(prep.info) > 1]
            exclude_preps = InferenceEngine.get_location_info(prepositions, ent, 0)
            assert all([len(prep.info) == 1 for prep in exclude_preps])
            exclude_locs = set([prep.info[0] for prep in exclude_preps])
            if maybe_preps:
                assert not set(maybe_preps[0].info).intersection(exclude_locs)
            if sure_preps:
                assert not set(sure_preps[0].info).intersection(exclude_locs)
            #
            # coloc_preps = InferenceEngine.get_coloc_info(prepositions, ent)
            if ent in held_objects:
                # one holder for every held object
                holders = InferenceEngine.get_possesion_info(prepositions, obj=ent,
                                                             confidence=1)
                assert len(holders) == 1


def solve_inference_engine(world, return_engine: bool = False):
    answers = {}
    supp_facts = {}
    engine = InferenceEngine()
    last_q = list(world.q_history.keys())[-1]
    for ind in range(1, last_q+1):
        # print(ind)
        if ind in world.history:
            engine.add_event(world.history[ind])
        else:
            ans, s_facts = engine.solve_question(world.q_history[ind])
            answers[ind] = ans
            supp_facts[ind] = s_facts
    if return_engine:
        return answers, supp_facts, engine
    else:
        return answers, supp_facts
