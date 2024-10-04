"""
This module translates incoming feedback of different types into a common format.
"""

from rlhfblender.data_models.feedback_models import (
    AbsoluteFeedback,
    Actuality,
    Content,
    Description,
    Evaluation,
    FeedbackType,
    Granularity,
    Instruction,
    Intention,
    Relation,
    RelativeEvaluation,
    RelativeFeedback,
    RelativeInstruction,
    StandardizedFeedback,
    StandardizedFeedbackType,
    UnprocessedFeedback,    
    get_granularity,
    get_target,
    # TextFeedback,
    get_text_feedback,
    get_feedback_type,
)
from rlhfblender.data_models.global_models import Environment, Experiment
from rlhfblender.logger.csv_logger import CSVLogger
from rlhfblender.logger.json_logger import JSONLogger


class FeedbackTranslator:
    """
    This class translates incoming feedback of different types into a common format (StandardizedFeedback).

    : param experiment: The experiment object
    : param env: The environment object
    """

    def __init__(self, experiment: Experiment, env: Environment):
        self.experiment = experiment
        self.env = env

        self.feedback_id = 0

        self.logger = JSONLogger(experiment, env, "feedback") if experiment is not None and env is not None else None
        self.feedback_buffer = []
        self.feedback_cache = {}
    
    async def cache_feedback(self, episode_id: str, feedback: str):
        if episode_id not in self.feedback_cache:
            self.feedback_cache[episode_id] = []
        self.feedback_cache[episode_id].append(feedback)

    def get_cached_feedback(self, episode_id: str):
        return self.feedback_cache.get(episode_id, [])
    
    async def get_all_cached_feedback(self):
        return self.feedback_cache 


    def set_translator(self, experiment: Experiment, env: Environment) -> str:
        """
        Sets the experiment and environment for the translator
        :param experiment: The experiment object
        :param env: The environment object
        :return: The logger ID
        """
        self.experiment = experiment
        self.env = env

        self.logger = CSVLogger(experiment, env, "feedback")

        self.reset()

        return self.logger.logger_id

    def reset(self) -> None:
        """
        Resets the feedback translator
        :return:
        """
        self.feedback_id = 0
        self.logger.reset()
        self.feedback_buffer = []
    
    @classmethod
    def map_feedback_type_to_standardized(cls,feedback_type):
        mapping = {
            "Critique": {
                "intention": Intention.evaluate,
                "actuality": Actuality.observed,
                "relation": Relation.absolute,
                "content": Content.feature,
            },
            "Suggestion": {
                "intention": Intention.instruct,
                "actuality": Actuality.observed,
                "relation": Relation.absolute,
                "content": Content.instance,
            },
            "Observation": {
                "intention": Intention.describe,
                "actuality": Actuality.observed,
                "relation": Relation.absolute,
                "content": Content.instance,
            },
            "Comparison": {
                "intention": Intention.evaluate,
                "actuality": Actuality.observed,
                "relation": Relation.relative,
                "content": Content.feature,
            },
            "Mission": {
                "intention": Intention.instruct,
                "actuality": Actuality.hypothetical,
                "relation": Relation.absolute,
                "content": Content.feature,
            },
            "Prioritization": {
                "intention": Intention.instruct,
                "actuality": Actuality.observed,
                "relation": Relation.relative,
                "content": Content.feature,
            },
            "Miscellaneous": {
                "intention": Intention.none,
                "actuality": Actuality.hypothetical,
                "relation": Relation.absolute,
                "content": Content.feature,
            },            
        }
        return mapping.get(feedback_type, mapping["Critique"])

    @classmethod
    def get_content_type(cls,txt_feed_type):
        goal_preferences = [
            {"goal": item["goal"], "priority": item["priority"]}
            for item in txt_feed_type["goal_preferences"]
        ]                
        if txt_feed_type["category"] == "Critique":
            return Evaluation(score=txt_feed_type["score"])
        elif txt_feed_type["category"] == "Suggestion":
            # goal = txt_feed_type["goal"] if isinstance(txt_feed_type["goal"], dict) else txt_feed_type["goal"]
            return Instruction(action=txt_feed_type["action"], goal=txt_feed_type["goal"])
        elif txt_feed_type["category"] == "Observation":
            return Description(feature_selection=txt_feed_type["feature_selection"], feature_importance=txt_feed_type["feature_importance"])
        elif txt_feed_type["category"] == "Comparison":
            return RelativeEvaluation(preferences=txt_feed_type["preferences"])
        elif txt_feed_type["category"] == "Mission":
            goal = txt_feed_type["goal"] if isinstance(txt_feed_type["goal"], dict) else txt_feed_type["goal"]
            return Instruction(action=txt_feed_type["action"], goal=goal)
        elif txt_feed_type["category"] == "Prioritization":
            return RelativeInstruction(action_preferences=txt_feed_type["action_preferences"], goal_preferences=goal_preferences)
        elif txt_feed_type["category"] == "Miscellaneous":
            return Evaluation(score=txt_feed_type["score"])
        else:
            return Evaluation(score=txt_feed_type["score"])


    async def give_feedback(self, session_id: str, feedback: UnprocessedFeedback) -> StandardizedFeedback:
        """
        We get either a single number or a list of numbers as feedback. We need to translate this into a common format
        called StandardizedFeedback
        :param session_id: The session ID
        :param feedback: (UnprocessedFeedback) The feedback
        :return: (StandardizedFeedback) The standardized feedback
        """
        return_feedback = None
        print("this is the initial feedback text : ", feedback.textFeedback)
        if feedback.feedback_type == FeedbackType.rating:
            return_feedback = AbsoluteFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.evaluate,
                    actuality=Actuality.observed,
                    relation=Relation.absolute,
                    content=Content.instance,
                    granularity=get_granularity(feedback.granularity),
                ),
                target=get_target(feedback.targets[0], feedback.granularity),
                content=Evaluation(score=feedback.score),
            )
        elif feedback.feedback_type == FeedbackType.ranking:
            return_feedback = RelativeFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.evaluate,
                    actuality=Actuality.observed,
                    relation=Relation.relative,
                    content=Content.instance,
                    granularity=Granularity.episode,
                ),
                target=[get_target(target, feedback.granularity) for target in feedback.targets],  # is a list in this case
                content=RelativeEvaluation(preferences=feedback.preferences),
            )
        elif feedback.feedback_type == FeedbackType.correction:
            return_feedback = RelativeFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.instruct,
                    actuality=Actuality.observed,
                    relation=Relation.relative,
                    content=Content.instance,
                    granularity=Granularity.state,
                ),
                target=[get_target(target, feedback.granularity) for target in feedback.targets],  # is a list in this case
                content=RelativeInstruction(action_preferences=feedback.action_preferences),
            )
        elif feedback.feedback_type == FeedbackType.demonstration:
            return_feedback = AbsoluteFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.instruct,
                    actuality=Actuality.hypothetical,
                    relation=Relation.absolute,
                    content=Content.instance,
                    granularity=Granularity.state,
                ),
                target=get_target(feedback.targets[0], feedback.granularity),
                content=Instruction(action=[]),  # Content is already in the target (i.e. states and actions)
            )
        elif feedback.feedback_type == FeedbackType.featureSelection:
            return_feedback = AbsoluteFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=Intention.describe,
                    actuality=Actuality.observed,
                    relation=Relation.absolute,
                    content=Content.instance,
                    granularity=Granularity.entire,
                ),
                target=get_target(feedback.targets[0], feedback.granularity),
                content=Description(feature_selection=feedback.feature_selection),
            )
        elif feedback.feedback_type == FeedbackType.textual:
            current_episode_id = feedback.targets[0]['target_id']
            await self.cache_feedback(current_episode_id, feedback.textFeedback)
            all_feedback = await self.get_all_cached_feedback()            
            # print('textual feedback is running')
            # print(feedback.textFeedback)
            final_feedback = await get_text_feedback(feedback.textFeedback,all_feedback)
            feedback_type_mapping = self.map_feedback_type_to_standardized(final_feedback["category"])                   
            return_feedback = AbsoluteFeedback(
                feedback_id=self.feedback_id,
                feedback_timestamp=feedback.timestamp,
                feedback_type=StandardizedFeedbackType(
                    intention=feedback_type_mapping["intention"],
                    actuality=feedback_type_mapping["actuality"],
                    relation=feedback_type_mapping["relation"],
                    content=feedback_type_mapping["content"],
                    granularity= Granularity.episode,
                    txt_feedback_type = get_feedback_type(final_feedback["category"]),
                    txt_score = final_feedback["score"],
                    txt_feedback= feedback.textFeedback,                                    
                ),
                target=get_target(feedback.targets[0], feedback.granularity),
                content= self.get_content_type(final_feedback)
            )

        self.feedback_id += 1

        self.logger.log_raw(feedback)        

        self.feedback_buffer.append(return_feedback)

    def submit(self, session_id: str,) -> None:
        """
        Submits the content of the current feedback buffer to the feedback dataset
        :param session_id: The session ID
        :return: None
        """
        # De-duplicate feedback in the feedback buffer
        # If feedback.episode_id and feedback.feedback_type are the same, we can assume that the feedback is the same,
        # we just want to keep the latest one
        feedback_dict = {}
        for feedback in self.feedback_buffer:
            if isinstance(feedback, AbsoluteFeedback):
                feedback_dict[(feedback.target.target_id, feedback.feedback_type)] = feedback
            else:
                feedback_dict[(feedback.target[0].target_id, feedback.feedback_type)] = feedback

        self.feedback_buffer = list(feedback_dict.values())

        for feedback in self.feedback_buffer:
            self.logger.log(feedback)       

        self.feedback_buffer = []
