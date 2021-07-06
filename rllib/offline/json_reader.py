import glob
import json
import logging
import os
import random
from urllib.parse import urlparse

from ray.rllib.utils.debug import summarize
try:
    from smart_open import smart_open
except ImportError:
    smart_open = None

from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch, \
    SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.compression import unpack_if_needed
from ray.rllib.utils.typing import FileType, SampleBatchType
from typing import List

logger = logging.getLogger(__name__)

WINDOWS_DRIVES = [chr(i) for i in range(ord("c"), ord("z") + 1)]


@PublicAPI
class JsonReader(InputReader):
    """Reader object that loads experiences from JSON file chunks.

    The input files will be read from in an random order."""

    @PublicAPI
    def __init__(self, inputs: List[str], ioctx: IOContext = None):
        """Initialize a JsonReader.

        Args:
            inputs (str|list): either a glob expression for files, e.g.,
                "/tmp/**/*.json", or a list of single file paths or URIs, e.g.,
                ["s3://bucket/file.json", "s3://bucket/file2.json"].
            ioctx (IOContext): current IO context object.
        """

        self.ioctx = ioctx or IOContext()
        self.default_policy = None
        if self.ioctx.worker is not None:
            self.default_policy = \
                self.ioctx.worker.policy_map.get(DEFAULT_POLICY_ID)
        if isinstance(inputs, str):
            inputs = os.path.abspath(os.path.expanduser(inputs))
            if os.path.isdir(inputs):
                inputs = os.path.join(inputs, "*.json")
                logger.warning(
                    "Treating input directory as glob pattern: {}".format(
                        inputs))
            if urlparse(inputs).scheme not in [""] + WINDOWS_DRIVES:
                raise ValueError(
                    "Don't know how to glob over `{}`, ".format(inputs) +
                    "please specify a list of files to read instead.")
            else:
                self.files = glob.glob(inputs)
        elif type(inputs) is list:
            self.files = inputs
        else:
            raise ValueError(
                "type of inputs must be list or str, not {}".format(inputs))
        if self.files:
            logger.info("Found {} input files.".format(len(self.files)))
        else:
            raise ValueError("No files found matching {}".format(inputs))
        self.cur_file = None

    @override(InputReader)
    def next(self) -> SampleBatchType:
        batch = self._try_parse(self._next_line())
        tries = 0
        while not batch and tries < 100:
            tries += 1
            logger.debug("Skipping empty line in {}".format(self.cur_file))
            batch = self._try_parse(self._next_line())
        if not batch:
            raise ValueError(
                "Failed to read valid experience batch from file: {}".format(
                    self.cur_file))
        return self._postprocess_if_needed(batch)

    def _postprocess_if_needed(self,
                               batch: SampleBatchType) -> SampleBatchType:
        if not self.ioctx.config.get("postprocess_inputs"):
            return batch

        if isinstance(batch, SampleBatch):
            out = []
            for sub_batch in batch.split_by_episode():
                out.append(
                    self.default_policy.postprocess_trajectory(sub_batch))
            return SampleBatch.concat_samples(out)
        elif isinstance(batch, MultiAgentBatch):

            from ray.rllib.evaluation.episode import MultiAgentEpisode
            from ray.rllib.evaluation.sample_batch_builder import \
                MultiAgentSampleBatchBuilder
            from ray.rllib.evaluation.collectors.simple_list_collector import \
                SimpleListCollector
            sample_collector = SimpleListCollector(
                self.ioctx.worker.policy_map,
                False,
                self.ioctx.config.get("callbacks")(),
                False,
                self.ioctx.config.get("rollout_fragment_length"),
                count_steps_by=self.ioctx.config.get("multiagent")["count_steps_by"])
            def get_batch_builder():
                return None
            def new_episode(env_id):
                episode = MultiAgentEpisode(
                    self.ioctx.worker.policy_map,
                    self.ioctx.config.get("multiagent")["policy_mapping_fn"],
                    get_batch_builder,
                    self.default_policy,
                    env_id=env_id)
                return episode

            active_episode = new_episode(0)
            # logger.info(summarize())
            # 应对多智能体的问题。就是两个策略批次。
            for policy_id, policy_batch in batch.policy_batches.items():
                # logger.info(policy_id)
                for i in range(batch.env_steps()):
                    if i==0:
                        sample_collector.add_init_obs(active_episode, policy_batch.data["agent_index"][i], 0,
                                                    policy_id, -1,
                                                    policy_batch.data["obs"][0])
                        continue

                    # logger.info(MultiAgentBatch)
                    values_dict = {
                        "t": i-1,
                        "env_id": 0,
                        "agent_index": policy_batch.data["agent_index"][i],
                        # Action (slot 0) taken at timestep t.
                        "actions": policy_batch.data["actions"][i-1],
                        # Reward received after taking a at timestep t.
                        "rewards": policy_batch.data["rewards"][i],
                        # After taking action=a, did we reach terminal?
                        "dones": policy_batch.data["dones"][i],
                        # Next observation.
                        "new_obs": policy_batch.data["obs"][i],
                    }
                    sample_collector.add_action_reward_next_obs(
                        active_episode.episode_id, policy_batch.data["agent_index"][i], 0, policy_id,
                        policy_batch.data["dones"][i], values_dict)
            postprocessed_batch = sample_collector.postprocess_episode(
                active_episode,
                is_done=True,
                check_dones=False,
                build=True)
            return postprocessed_batch

    def _try_parse(self, line: str) -> SampleBatchType:
        line = line.strip()
        if not line:
            return None
        try:
            return _from_json(line)
        except Exception:
            logger.exception("Ignoring corrupt json record in {}: {}".format(
                self.cur_file, line))
            return None

    def _next_line(self) -> str:
        if not self.cur_file:
            self.cur_file = self._next_file()
        line = self.cur_file.readline()
        tries = 0
        while not line and tries < 100:
            tries += 1
            if hasattr(self.cur_file, "close"):  # legacy smart_open impls
                self.cur_file.close()
            self.cur_file = self._next_file()
            line = self.cur_file.readline()
            if not line:
                logger.debug("Ignoring empty file {}".format(self.cur_file))
        if not line:
            raise ValueError("Failed to read next line from files: {}".format(
                self.files))
        return line

    def _next_file(self) -> FileType:
        path = random.choice(self.files)
        if urlparse(path).scheme not in [""] + WINDOWS_DRIVES:
            if smart_open is None:
                raise ValueError(
                    "You must install the `smart_open` module to read "
                    "from URIs like {}".format(path))
            return smart_open(path, "r")
        else:
            return open(path, "r")


def _from_json(batch: str) -> SampleBatchType:
    if isinstance(batch, bytes):  # smart_open S3 doesn't respect "r"
        batch = batch.decode("utf-8")
    data = json.loads(batch)

    if "type" in data:
        data_type = data.pop("type")
    else:
        raise ValueError("JSON record missing 'type' field")

    if data_type == "SampleBatch":
        for k, v in data.items():
            data[k] = unpack_if_needed(v)
        return SampleBatch(data)
    elif data_type == "MultiAgentBatch":
        policy_batches = {}
        for policy_id, policy_batch in data["policy_batches"].items():
            inner = {}
            for k, v in policy_batch.items():
                inner[k] = unpack_if_needed(v)
            policy_batches[policy_id] = SampleBatch(inner)
        return MultiAgentBatch(policy_batches, data["count"])
    else:
        raise ValueError(
            "Type field must be one of ['SampleBatch', 'MultiAgentBatch']",
            data_type)
