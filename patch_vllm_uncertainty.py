#!/usr/bin/env python3
"""Patch vLLM 0.26.0 to return token uncertainty and LM-head variance.

This is a single-file patcher for Linux vLLM environments, including Vast.ai.
It patches the installed vLLM package used by the Python interpreter that runs
this script, creates byte-for-byte backups, validates exact upstream hashes,
and can safely revert the change.

Response policy:
* non-streaming: ``uncertainty.variance`` is the arithmetic mean of per-token
  variances, where variance = exp(log_variance).
* streaming: every generated-token SSE chunk contains the variance and
  log-variance of the last token represented by that chunk.
* request ``return_token_variances: true`` to include per-token variance and
  log-variance lists.

The uncertainty head is loaded from ``VLLM_UNCERTAINTY_HEAD_PATH`` or, when
that variable is absent, ``<model>/heads/log_variance_head.safetensors``.

LM-head variance is an independent, opt-in feature:
* start ``vllm serve`` with ``--enable-lm-head-variance``;
* request ``return_lm_head_variance: true`` for an aggregate;
* request ``return_token_lm_head_variances: true`` for the non-streaming list.

It is the population variance across the raw LM-head vocabulary logits for
each generated token. It neither reads nor changes the learned uncertainty
head. Streaming returns the current chunk's last-token value; non-streaming
returns the mean, optionally with the per-token list.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import py_compile
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PATCH_ID = "aigen-uncertainty-vllm-0.26.0-v4"
LEGACY_PATCH_IDS = {"aigen-uncertainty-vllm-0.26.0-v3"}
SUPPORTED_VLLM_VERSION = "0.26.0"
BACKUP_DIRNAME = ".aigen_uncertainty_backup_vllm_0_26_0"
MANIFEST_NAME = "manifest.json"


class PatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Replacement:
    old: str
    new: str
    expected: int = 1


@dataclass(frozen=True)
class FilePatch:
    relative_path: str
    replacements: tuple[Replacement, ...]


EXPECTED_SHA256 = {
    "entrypoints/openai/cli_args.py": "c0deeb051627e1eb54d0b4bb00cf0f751d55729c3ff41377cef58d9c48d9d129",
    "model_executor/models/mistral3.py": "49be7412b7dc20c0b47e68f872f562a105fc106b0b64fb10ddc387d6e01f489c",
    "v1/outputs.py": "1e87bf44162452c1908d3a5003685937dbdc56f5634e35e11ed7b6a5322a1c15",
    "v1/worker/gpu_model_runner.py": "81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0",
    "v1/worker/gpu/model_runner.py": "77c8042cd911fbb0afc5d7a731a293afd7d2465d9ceea9574a69dab5ee5182be",
    "v1/worker/gpu/async_utils.py": "033e176124773191c6d8a5b699e8cc6475a46498e9ea89d7b595ee8857bf4d2c",
    "v1/engine/__init__.py": "79aa7c299953c0d6133d9112cf24d373bd169e884ab27322a6590cd8577eedad",
    "v1/core/sched/scheduler.py": "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941",
    "outputs.py": "44e2ea5d8a34403a99a867b42806e8102b3d9e7b1a0f735af26e6650cc482342",
    "v1/engine/output_processor.py": "ee10351275d90796c8b901a5f4b23d5a046ef6ee72fd2921aff2ae78ca58bd9b",
    "entrypoints/openai/chat_completion/protocol.py": "aea283940efbe7b8c91aed6e4e146a9d7f714095cf46c6df3080f00e05d3277c",
    "entrypoints/openai/chat_completion/serving.py": "860e27d548ec7bb0497a46e0356a5acc5500998944cffa5e67e336041693c067",
}


PATCHES = (
    FilePatch(
        "entrypoints/openai/cli_args.py",
        (
            Replacement(
                "import argparse\n"
                "import json\n",
                "import argparse\n"
                "import json\n"
                "import os\n",
            ),
            Replacement(
                "    parser = FrontendArgs.add_cli_args(parser)\n",
                "    # AIGEN_UNCERTAINTY_PATCH:lm-head-server-option\n"
                "    parser.add_argument(\n"
                "        \"--enable-lm-head-variance\",\n"
                "        action=\"store_true\",\n"
                "        default=False,\n"
                "        help=(\n"
                "            \"Enable request-controlled population variance over \"\n"
                "            \"raw LM-head vocabulary logits.\"\n"
                "        ),\n"
                "    )\n"
                "    parser = FrontendArgs.add_cli_args(parser)\n",
            ),
            Replacement(
                "def validate_parsed_serve_args(args: argparse.Namespace):\n"
                "    \"\"\"Quick checks for model serve args that raise prior to loading.\"\"\"\n",
                "def validate_parsed_serve_args(args: argparse.Namespace):\n"
                "    \"\"\"Quick checks for model serve args that raise prior to loading.\"\"\"\n"
                "    # AIGEN_UNCERTAINTY_PATCH:lm-head-server-enable\n"
                "    if getattr(args, \"enable_lm_head_variance\", False):\n"
                "        os.environ[\"VLLM_ENABLE_LM_HEAD_VARIANCE\"] = \"1\"\n"
                "    else:\n"
                "        os.environ.pop(\"VLLM_ENABLE_LM_HEAD_VARIANCE\", None)\n",
            ),
        ),
    ),
    FilePatch(
        "model_executor/models/mistral3.py",
        (
            Replacement(
                "from collections.abc import Iterable, Mapping, Sequence\n"
                "from typing import Annotated, Literal\n",
                "from collections.abc import Iterable, Mapping, Sequence\n"
                "import os\n"
                "from pathlib import Path\n"
                "from typing import Annotated, Literal\n",
            ),
            Replacement(
                "import torch\n"
                "import torch.nn as nn\n",
                "import torch\n"
                "import torch.nn as nn\n"
                "from safetensors.torch import load_file as load_safetensors_file\n",
            ),
            Replacement(
                "        with self._mark_language_model(vllm_config):\n"
                "            self.language_model = init_vllm_registered_model(\n"
                "                vllm_config=vllm_config,\n"
                "                hf_config=config.text_config,\n"
                "                prefix=maybe_prefix(prefix, \"language_model\"),\n"
                "            )\n\n"
                "        self.make_empty_intermediate_tensors = (\n",
                "        with self._mark_language_model(vllm_config):\n"
                "            self.language_model = init_vllm_registered_model(\n"
                "                vllm_config=vllm_config,\n"
                "                hf_config=config.text_config,\n"
                "                prefix=maybe_prefix(prefix, \"language_model\"),\n"
                "            )\n\n"
                "        # AIGEN_UNCERTAINTY_PATCH:model-init\n"
                "        configured_head = os.environ.get(\"VLLM_UNCERTAINTY_HEAD_PATH\")\n"
                "        default_head = (\n"
                "            Path(str(vllm_config.model_config.model))\n"
                "            / \"heads\"\n"
                "            / \"log_variance_head.safetensors\"\n"
                "        )\n"
                "        head_path = Path(configured_head) if configured_head else default_head\n"
                "        self._uncertainty_head_path: Path | None = None\n"
                "        self.log_variance_head: nn.Linear | None = None\n"
                "        if head_path.is_file():\n"
                "            self._uncertainty_head_path = head_path\n"
                "            self.log_variance_head = nn.Linear(\n"
                "                int(config.text_config.hidden_size), 1, bias=True\n"
                "            )\n"
                "        elif configured_head:\n"
                "            raise FileNotFoundError(\n"
                "                f\"VLLM_UNCERTAINTY_HEAD_PATH does not exist: {head_path}\"\n"
                "            )\n\n"
                "        self.make_empty_intermediate_tensors = (\n",
            ),
            Replacement(
                "    def compute_logits(\n"
                "        self,\n"
                "        hidden_states: torch.Tensor,\n"
                "    ) -> torch.Tensor | None:\n"
                "        return self.language_model.compute_logits(hidden_states)\n\n"
                "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n",
                "    def compute_logits(\n"
                "        self,\n"
                "        hidden_states: torch.Tensor,\n"
                "    ) -> torch.Tensor | None:\n"
                "        return self.language_model.compute_logits(hidden_states)\n\n"
                "    # AIGEN_UNCERTAINTY_PATCH:model-compute\n"
                "    def compute_uncertainty(\n"
                "        self, hidden_states: torch.Tensor\n"
                "    ) -> torch.Tensor | None:\n"
                "        if self.log_variance_head is None:\n"
                "            return None\n"
                "        return self.log_variance_head(hidden_states).reshape(-1)\n\n"
                "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n",
            ),
            Replacement(
                "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
                "        loader = AutoWeightsLoader(self)\n"
                "        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n",
                "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
                "        loader = AutoWeightsLoader(self)\n"
                "        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n"
                "        # AIGEN_UNCERTAINTY_PATCH:model-load\n"
                "        if self.log_variance_head is not None:\n"
                "            assert self._uncertainty_head_path is not None\n"
                "            head_tensors = load_safetensors_file(\n"
                "                str(self._uncertainty_head_path), device=\"cpu\"\n"
                "            )\n"
                "            weight = head_tensors.get(\"log_variance_head.weight\")\n"
                "            bias = head_tensors.get(\"log_variance_head.bias\")\n"
                "            if weight is None or bias is None:\n"
                "                raise KeyError(\n"
                "                    \"uncertainty head must contain log_variance_head.weight \"\n"
                "                    \"and log_variance_head.bias\"\n"
                "                )\n"
                "            if tuple(weight.shape) != tuple(self.log_variance_head.weight.shape):\n"
                "                raise ValueError(\n"
                "                    f\"uncertainty weight shape mismatch: {tuple(weight.shape)}\"\n"
                "                )\n"
                "            if tuple(bias.shape) != tuple(self.log_variance_head.bias.shape):\n"
                "                raise ValueError(\n"
                "                    f\"uncertainty bias shape mismatch: {tuple(bias.shape)}\"\n"
                "                )\n"
                "            with torch.no_grad():\n"
                "                self.log_variance_head.weight.copy_(\n"
                "                    weight.to(\n"
                "                        device=self.log_variance_head.weight.device,\n"
                "                        dtype=self.log_variance_head.weight.dtype,\n"
                "                    )\n"
                "                )\n"
                "                self.log_variance_head.bias.copy_(\n"
                "                    bias.to(\n"
                "                        device=self.log_variance_head.bias.device,\n"
                "                        dtype=self.log_variance_head.bias.dtype,\n"
                "                    )\n"
                "                )\n"
                "            loaded.update(\n"
                "                {\"log_variance_head.weight\", \"log_variance_head.bias\"}\n"
                "            )\n"
                "        return loaded\n",
            ),
        ),
    ),
    FilePatch(
        "v1/outputs.py",
        (
            Replacement(
                "    # req_id -> num_nans_in_logits\n"
                "    num_nans_in_logits: dict[str, int] | None = None\n",
                "    # req_id -> num_nans_in_logits\n"
                "    num_nans_in_logits: dict[str, int] | None = None\n\n"
                "    # AIGEN_UNCERTAINTY_PATCH:model-runner-output\n"
                "    # num_reqs x num_generated_tokens\n"
                "    uncertainty_log_variances: list[list[float]] | None = None\n"
                "    # Independent population variance over raw LM-head logits.\n"
                "    lm_head_variances: list[list[float]] | None = None\n",
            ),
        ),
    ),
    FilePatch(
        "v1/worker/gpu_model_runner.py",
        (
            Replacement(
                "import itertools\n"
                "import threading\n",
                "import itertools\n"
                "import os\n"
                "import threading\n",
            ),
            Replacement(
                "        # Apply structured output bitmasks if present.\n",
                "        # AIGEN_UNCERTAINTY_PATCH:legacy-lm-head-compute\n"
                "        lm_head_variances: torch.Tensor | None = None\n"
                "        lm_head_variance_requested = any(\n"
                "            bool(\n"
                "                self.requests[req_id].sampling_params.extra_args.get(\n"
                "                    \"return_lm_head_variance\", False\n"
                "                )\n"
                "            )\n"
                "            for req_id in self.input_batch.req_ids\n"
                "            if self.requests[req_id].sampling_params is not None\n"
                "            and self.requests[req_id].sampling_params.extra_args\n"
                "        )\n"
                "        if (\n"
                "            os.getenv(\"VLLM_ENABLE_LM_HEAD_VARIANCE\") == \"1\"\n"
                "            and lm_head_variance_requested\n"
                "        ):\n"
                "            lm_head_variances = torch.var(\n"
                "                logits.detach().float(), dim=-1, correction=0\n"
                "            )\n\n"
                "        # Apply structured output bitmasks if present.\n",
            ),
            Replacement(
                "        with record_function_or_nullcontext(\"gpu_model_runner: sample\"):\n"
                "            sampler_output = self._sample(logits, spec_decode_metadata)\n\n"
                "        self._update_states_after_model_execute(\n",
                "        with record_function_or_nullcontext(\"gpu_model_runner: sample\"):\n"
                "            sampler_output = self._sample(logits, spec_decode_metadata)\n\n"
                "        # AIGEN_UNCERTAINTY_PATCH:legacy-gpu-compute\n"
                "        uncertainty_log_variances: torch.Tensor | None = None\n"
                "        compute_uncertainty = getattr(self.model, \"compute_uncertainty\", None)\n"
                "        if callable(compute_uncertainty):\n"
                "            uncertainty_tensor = compute_uncertainty(sample_hidden_states)\n"
                "            if uncertainty_tensor is not None:\n"
                "                uncertainty_log_variances = (\n"
                "                    uncertainty_tensor.detach().float()\n"
                "                )\n\n"
                "        self._update_states_after_model_execute(\n",
            ),
            Replacement(
                "        with record_function_or_nullcontext(\"gpu_model_runner: ModelRunnerOutput\"):\n"
                "            output = ModelRunnerOutput(\n",
                "        # AIGEN_UNCERTAINTY_PATCH:legacy-gpu-map\n"
                "        per_request_log_variances: list[list[float]] | None = None\n"
                "        if (\n"
                "            uncertainty_log_variances is not None\n"
                "            and not self.use_async_scheduling\n"
                "        ):\n"
                "            uncertainty_log_variance_values = (\n"
                "                uncertainty_log_variances.cpu().tolist()\n"
                "            )\n"
                "            if len(uncertainty_log_variance_values) != len(\n"
                "                valid_sampled_token_ids\n"
                "            ):\n"
                "                raise RuntimeError(\n"
                "                    \"uncertainty batch does not match sampled-token batch\"\n"
                "                )\n"
                "            per_request_log_variances = []\n"
                "            for value, token_ids in zip(\n"
                "                uncertainty_log_variance_values, valid_sampled_token_ids\n"
                "            ):\n"
                "                if len(token_ids) > 1:\n"
                "                    raise RuntimeError(\n"
                "                        \"uncertainty patch does not support speculative \"\n"
                "                        \"decoding; disable speculative_config\"\n"
                "                    )\n"
                "                per_request_log_variances.append(\n"
                "                    [float(value)] if token_ids else []\n"
                "                )\n\n"
                "        per_request_lm_head_variances: list[list[float]] | None = None\n"
                "        if lm_head_variances is not None and not self.use_async_scheduling:\n"
                "            lm_head_variance_values = lm_head_variances.cpu().tolist()\n"
                "            if len(lm_head_variance_values) != len(\n"
                "                valid_sampled_token_ids\n"
                "            ):\n"
                "                raise RuntimeError(\n"
                "                    \"LM-head variance batch does not match sampled-token batch\"\n"
                "                )\n"
                "            per_request_lm_head_variances = []\n"
                "            for value, token_ids in zip(\n"
                "                lm_head_variance_values, valid_sampled_token_ids\n"
                "            ):\n"
                "                if len(token_ids) > 1:\n"
                "                    raise RuntimeError(\n"
                "                        \"LM-head variance does not support speculative \"\n"
                "                        \"decoding; disable speculative_config\"\n"
                "                    )\n"
                "                per_request_lm_head_variances.append(\n"
                "                    [float(value)] if token_ids else []\n"
                "                )\n\n"
                "        with record_function_or_nullcontext(\"gpu_model_runner: ModelRunnerOutput\"):\n"
                "            output = ModelRunnerOutput(\n",
            ),
            Replacement(
                "                sampled_token_ids=valid_sampled_token_ids,\n"
                "                logprobs=logprobs_lists,\n",
                "                sampled_token_ids=valid_sampled_token_ids,\n"
                "                uncertainty_log_variances=per_request_log_variances,\n"
                "                lm_head_variances=per_request_lm_head_variances,\n"
                "                logprobs=logprobs_lists,\n",
            ),
            Replacement(
                "        logprobs_tensors: LogprobsTensors | None,\n"
                "        invalid_req_indices: list[int],\n",
                "        logprobs_tensors: LogprobsTensors | None,\n"
                "        uncertainty_log_variances: torch.Tensor | None,\n"
                "        lm_head_variances: torch.Tensor | None,\n"
                "        invalid_req_indices: list[int],\n",
            ),
            Replacement(
                "        self._logprobs_tensors = logprobs_tensors\n"
                "        self._routed_experts = routed_experts\n",
                "        self._logprobs_tensors = logprobs_tensors\n"
                "        # AIGEN_UNCERTAINTY_PATCH:legacy-async-retain\n"
                "        self._uncertainty_log_variances = uncertainty_log_variances\n"
                "        self._lm_head_variances = lm_head_variances\n"
                "        self._routed_experts = routed_experts\n",
            ),
            Replacement(
                "            self._logprobs_tensors_cpu = (\n"
                "                self._logprobs_tensors.to_cpu_nonblocking()\n"
                "                if self._logprobs_tensors\n"
                "                else None\n"
                "            )\n"
                "            self._routed_experts_cpu = (\n",
                "            self._logprobs_tensors_cpu = (\n"
                "                self._logprobs_tensors.to_cpu_nonblocking()\n"
                "                if self._logprobs_tensors\n"
                "                else None\n"
                "            )\n"
                "            # AIGEN_UNCERTAINTY_PATCH:legacy-async-copy\n"
                "            self._uncertainty_log_variances_cpu = (\n"
                "                self._uncertainty_log_variances.to(\n"
                "                    \"cpu\", non_blocking=True\n"
                "                )\n"
                "                if self._uncertainty_log_variances is not None\n"
                "                else None\n"
                "            )\n"
                "            self._lm_head_variances_cpu = (\n"
                "                self._lm_head_variances.to(\"cpu\", non_blocking=True)\n"
                "                if self._lm_head_variances is not None\n"
                "                else None\n"
                "            )\n"
                "            self._routed_experts_cpu = (\n",
            ),
            Replacement(
                "        del self._logprobs_tensors\n"
                "        del self._sampled_token_ids\n",
                "        del self._logprobs_tensors\n"
                "        del self._sampled_token_ids\n"
                "        del self._uncertainty_log_variances\n"
                "        del self._lm_head_variances\n",
            ),
            Replacement(
                "        output.sampled_token_ids = valid_sampled_token_ids\n"
                "        output.logprobs = logprobs_lists\n\n"
                "        if self._routed_experts_cpu is not None:\n",
                "        output.sampled_token_ids = valid_sampled_token_ids\n"
                "        output.logprobs = logprobs_lists\n\n"
                "        # AIGEN_UNCERTAINTY_PATCH:legacy-async-map\n"
                "        if self._uncertainty_log_variances_cpu is not None:\n"
                "            values = (\n"
                "                self._uncertainty_log_variances_cpu.reshape(-1).tolist()\n"
                "            )\n"
                "            if len(values) != len(valid_sampled_token_ids):\n"
                "                raise RuntimeError(\n"
                "                    \"uncertainty batch does not match sampled-token batch\"\n"
                "                )\n"
                "            per_request: list[list[float]] = []\n"
                "            for value, token_ids in zip(values, valid_sampled_token_ids):\n"
                "                if len(token_ids) > 1:\n"
                "                    raise RuntimeError(\n"
                "                        \"uncertainty patch does not support speculative \"\n"
                "                        \"decoding; disable speculative_config\"\n"
                "                    )\n"
                "                per_request.append([float(value)] if token_ids else [])\n"
                "            output.uncertainty_log_variances = per_request\n\n"
                "        if self._lm_head_variances_cpu is not None:\n"
                "            values = self._lm_head_variances_cpu.reshape(-1).tolist()\n"
                "            if len(values) != len(valid_sampled_token_ids):\n"
                "                raise RuntimeError(\n"
                "                    \"LM-head variance batch does not match sampled-token batch\"\n"
                "                )\n"
                "            per_request_lm: list[list[float]] = []\n"
                "            for value, token_ids in zip(values, valid_sampled_token_ids):\n"
                "                if len(token_ids) > 1:\n"
                "                    raise RuntimeError(\n"
                "                        \"LM-head variance does not support speculative \"\n"
                "                        \"decoding; disable speculative_config\"\n"
                "                    )\n"
                "                per_request_lm.append([float(value)] if token_ids else [])\n"
                "            output.lm_head_variances = per_request_lm\n\n"
                "        if self._routed_experts_cpu is not None:\n",
            ),
            Replacement(
                "                logprobs_tensors=sampler_output.logprobs_tensors,\n"
                "                invalid_req_indices=invalid_req_indices,\n",
                "                logprobs_tensors=sampler_output.logprobs_tensors,\n"
                "                uncertainty_log_variances=uncertainty_log_variances,\n"
                "                lm_head_variances=lm_head_variances,\n"
                "                invalid_req_indices=invalid_req_indices,\n",
            ),
        ),
    ),
    FilePatch(
        "v1/worker/gpu/model_runner.py",
        (
            Replacement(
                "import functools\n"
                "import gc\n",
                "import functools\n"
                "import gc\n"
                "import os\n",
            ),
            Replacement(
                "        self.input_buffers = InputBuffers(\n",
                "        # AIGEN_UNCERTAINTY_PATCH:v2-lm-head-request-state\n"
                "        self._lm_head_variance_req_ids: set[str] = set()\n"
                "        self.input_buffers = InputBuffers(\n",
            ),
            Replacement(
                "    def _remove_request(self, req_id: str) -> bool:\n"
                "        # Call model_state.remove_request *before* req_states.remove_request\n",
                "    def _remove_request(self, req_id: str) -> bool:\n"
                "        self._lm_head_variance_req_ids.discard(req_id)\n"
                "        # Call model_state.remove_request *before* req_states.remove_request\n",
            ),
            Replacement(
                "            sampling_params = new_req_data.sampling_params\n"
                "            self.req_states.add_request(\n",
                "            sampling_params = new_req_data.sampling_params\n"
                "            if (\n"
                "                sampling_params is not None\n"
                "                and sampling_params.extra_args\n"
                "                and sampling_params.extra_args.get(\n"
                "                    \"return_lm_head_variance\", False\n"
                "                )\n"
                "            ):\n"
                "                self._lm_head_variance_req_ids.add(req_id)\n"
                "            self.req_states.add_request(\n",
            ),
            Replacement(
                "    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:\n"
                "        sample_hidden_states = hidden_states[input_batch.logits_indices]\n"
                "        logits = self.model.compute_logits(sample_hidden_states)\n"
                "        if grammar_output is not None:\n",
                "    ) -> tuple[\n"
                "        SamplerOutput, torch.Tensor, torch.Tensor, torch.Tensor | None\n"
                "    ]:\n"
                "        sample_hidden_states = hidden_states[input_batch.logits_indices]\n"
                "        logits = self.model.compute_logits(sample_hidden_states)\n"
                "        # AIGEN_UNCERTAINTY_PATCH:v2-lm-head-compute\n"
                "        lm_head_variances: torch.Tensor | None = None\n"
                "        lm_head_variance_requested = any(\n"
                "            req_id in self._lm_head_variance_req_ids\n"
                "            for req_id in input_batch.req_ids\n"
                "        )\n"
                "        if (\n"
                "            os.getenv(\"VLLM_ENABLE_LM_HEAD_VARIANCE\") == \"1\"\n"
                "            and lm_head_variance_requested\n"
                "        ):\n"
                "            if input_batch.num_draft_tokens != 0:\n"
                "                raise RuntimeError(\n"
                "                    \"LM-head variance does not support speculative \"\n"
                "                    \"decoding; disable speculative_config\"\n"
                "                )\n"
                "            lm_head_variances = torch.var(\n"
                "                logits.detach().float(), dim=-1, correction=0\n"
                "            )\n"
                "        if grammar_output is not None:\n",
            ),
            Replacement(
                "        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected\n",
                "        return (\n"
                "            sampler_output,\n"
                "            sampler_output.num_sampled,\n"
                "            sampler_output.num_rejected,\n"
                "            lm_head_variances,\n"
                "        )\n",
            ),
            Replacement(
                "        sampler_output, num_sampled, num_rejected = self.sample(\n"
                "            hidden_states, input_batch, grammar_output\n"
                "        )\n\n"
                "        if self.pp_handler is not None:\n",
                "        (\n"
                "            sampler_output,\n"
                "            num_sampled,\n"
                "            num_rejected,\n"
                "            lm_head_variances,\n"
                "        ) = self.sample(hidden_states, input_batch, grammar_output)\n\n"
                "        # AIGEN_UNCERTAINTY_PATCH:v2-gpu-compute\n"
                "        uncertainty_log_variances = None\n"
                "        compute_uncertainty = getattr(self.model, \"compute_uncertainty\", None)\n"
                "        if callable(compute_uncertainty):\n"
                "            if input_batch.num_draft_tokens != 0:\n"
                "                raise RuntimeError(\n"
                "                    \"uncertainty patch does not support speculative \"\n"
                "                    \"decoding; disable speculative_config\"\n"
                "                )\n"
                "            uncertainty_log_variances = compute_uncertainty(\n"
                "                hidden_states[input_batch.logits_indices]\n"
                "            )\n"
                "            if uncertainty_log_variances is not None:\n"
                "                uncertainty_log_variances = (\n"
                "                    uncertainty_log_variances.detach().float()\n"
                "                )\n\n"
                "        if self.pp_handler is not None:\n",
            ),
            Replacement(
                "            num_sampled_tokens=num_sampled,\n"
                "            main_stream=self.main_stream,\n",
                "            num_sampled_tokens=num_sampled,\n"
                "            uncertainty_log_variances=uncertainty_log_variances,\n"
                "            lm_head_variances=lm_head_variances,\n"
                "            main_stream=self.main_stream,\n",
            ),
        ),
    ),
    FilePatch(
        "v1/worker/gpu/async_utils.py",
        (
            Replacement(
                "        num_sampled_tokens: torch.Tensor,\n"
                "        main_stream: torch.cuda.Stream,\n",
                "        num_sampled_tokens: torch.Tensor,\n"
                "        uncertainty_log_variances: torch.Tensor | None,\n"
                "        lm_head_variances: torch.Tensor | None,\n"
                "        main_stream: torch.cuda.Stream,\n",
            ),
            Replacement(
                "            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)\n"
                "            self.prompt_logprobs_dict = {\n",
                "            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)\n"
                "            # AIGEN_UNCERTAINTY_PATCH:v2-async-copy\n"
                "            self.uncertainty_log_variances_np = (\n"
                "                async_copy_to_np(uncertainty_log_variances)\n"
                "                if uncertainty_log_variances is not None\n"
                "                else None\n"
                "            )\n"
                "            self.lm_head_variances_np = (\n"
                "                async_copy_to_np(lm_head_variances)\n"
                "                if lm_head_variances is not None\n"
                "                else None\n"
                "            )\n"
                "            self.prompt_logprobs_dict = {\n",
            ),
            Replacement(
                "        self.model_runner_output.sampled_token_ids = sampled_token_ids\n\n"
                "        if self.num_nans is not None:\n",
                "        self.model_runner_output.sampled_token_ids = sampled_token_ids\n\n"
                "        # AIGEN_UNCERTAINTY_PATCH:v2-async-map\n"
                "        if self.uncertainty_log_variances_np is not None:\n"
                "            values = self.uncertainty_log_variances_np.reshape(-1).tolist()\n"
                "            if len(values) != len(sampled_token_ids):\n"
                "                raise RuntimeError(\n"
                "                    \"uncertainty batch does not match sampled-token batch\"\n"
                "                )\n"
                "            per_request: list[list[float]] = []\n"
                "            for value, token_ids in zip(values, sampled_token_ids):\n"
                "                if len(token_ids) > 1:\n"
                "                    raise RuntimeError(\n"
                "                        \"uncertainty patch does not support speculative \"\n"
                "                        \"decoding; disable speculative_config\"\n"
                "                    )\n"
                "                per_request.append([float(value)] if token_ids else [])\n"
                "            self.model_runner_output.uncertainty_log_variances = per_request\n\n"
                "        if self.lm_head_variances_np is not None:\n"
                "            values = self.lm_head_variances_np.reshape(-1).tolist()\n"
                "            if len(values) != len(sampled_token_ids):\n"
                "                raise RuntimeError(\n"
                "                    \"LM-head variance batch does not match sampled-token batch\"\n"
                "                )\n"
                "            per_request_lm: list[list[float]] = []\n"
                "            for value, token_ids in zip(values, sampled_token_ids):\n"
                "                if len(token_ids) > 1:\n"
                "                    raise RuntimeError(\n"
                "                        \"LM-head variance does not support speculative \"\n"
                "                        \"decoding; disable speculative_config\"\n"
                "                    )\n"
                "                per_request_lm.append([float(value)] if token_ids else [])\n"
                "            self.model_runner_output.lm_head_variances = per_request_lm\n\n"
                "        if self.num_nans is not None:\n",
            ),
        ),
    ),
    FilePatch(
        "v1/engine/__init__.py",
        (
            Replacement(
                "    new_logprobs: LogprobsLists | None = None\n"
                "    new_prompt_logprobs_tensors: LogprobsTensors | None = None\n",
                "    new_logprobs: LogprobsLists | None = None\n"
                "    # AIGEN_UNCERTAINTY_PATCH:engine-output\n"
                "    new_log_variances: list[float] | None = None\n"
                "    new_lm_head_variances: list[float] | None = None\n"
                "    new_prompt_logprobs_tensors: LogprobsTensors | None = None\n",
            ),
        ),
    ),
    FilePatch(
        "v1/core/sched/scheduler.py",
        (
            Replacement(
                "        sampled_token_ids = model_runner_output.sampled_token_ids\n"
                "        logprobs = model_runner_output.logprobs\n",
                "        sampled_token_ids = model_runner_output.sampled_token_ids\n"
                "        # AIGEN_UNCERTAINTY_PATCH:scheduler-read\n"
                "        uncertainty_log_variances = (\n"
                "            model_runner_output.uncertainty_log_variances\n"
                "        )\n"
                "        lm_head_variances = model_runner_output.lm_head_variances\n"
                "        logprobs = model_runner_output.logprobs\n",
            ),
            Replacement(
                "                        new_logprobs=new_logprobs,\n"
                "                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,\n",
                "                        new_logprobs=new_logprobs,\n"
                "                        # AIGEN_UNCERTAINTY_PATCH:scheduler-write\n"
                "                        new_log_variances=(\n"
                "                            uncertainty_log_variances[req_index][\n"
                "                                : len(new_token_ids)\n"
                "                            ]\n"
                "                            if uncertainty_log_variances is not None\n"
                "                            and new_token_ids\n"
                "                            else None\n"
                "                        ),\n"
                "                        new_lm_head_variances=(\n"
                "                            lm_head_variances[req_index][\n"
                "                                : len(new_token_ids)\n"
                "                            ]\n"
                "                            if lm_head_variances is not None\n"
                "                            and new_token_ids\n"
                "                            else None\n"
                "                        ),\n"
                "                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,\n",
            ),
        ),
    ),
    FilePatch(
        "outputs.py",
        (
            Replacement(
                "    lora_request: LoRARequest | None = None\n\n"
                "    def finished(self) -> bool:\n",
                "    lora_request: LoRARequest | None = None\n"
                "    # AIGEN_UNCERTAINTY_PATCH:completion-output\n"
                "    log_variances: GenericSequence[float] | None = None\n"
                "    lm_head_variances: GenericSequence[float] | None = None\n\n"
                "    def finished(self) -> bool:\n",
            ),
            Replacement(
                "                        if next_completion.logprobs:\n"
                "                            assert completion.logprobs is not None\n"
                "                            completion.logprobs.extend(next_completion.logprobs)  # type: ignore[arg-type]\n"
                "                        completion.cumulative_logprob = (\n",
                "                        if next_completion.logprobs:\n"
                "                            assert completion.logprobs is not None\n"
                "                            completion.logprobs.extend(next_completion.logprobs)  # type: ignore[arg-type]\n"
                "                        # AIGEN_UNCERTAINTY_PATCH:delta-merge\n"
                "                        if next_completion.log_variances:\n"
                "                            if completion.log_variances is None:\n"
                "                                completion.log_variances = []\n"
                "                            elif not isinstance(\n"
                "                                completion.log_variances, MutableSequence\n"
                "                            ):\n"
                "                                completion.log_variances = list(\n"
                "                                    completion.log_variances\n"
                "                                )\n"
                "                            completion.log_variances.extend(\n"
                "                                next_completion.log_variances\n"
                "                            )\n"
                "                        if next_completion.lm_head_variances:\n"
                "                            if completion.lm_head_variances is None:\n"
                "                                completion.lm_head_variances = []\n"
                "                            elif not isinstance(\n"
                "                                completion.lm_head_variances, MutableSequence\n"
                "                            ):\n"
                "                                completion.lm_head_variances = list(\n"
                "                                    completion.lm_head_variances\n"
                "                                )\n"
                "                            completion.lm_head_variances.extend(\n"
                "                                next_completion.lm_head_variances\n"
                "                            )\n"
                "                        completion.cumulative_logprob = (\n",
            ),
        ),
    ),
    FilePatch(
        "v1/engine/output_processor.py",
        (
            Replacement(
                "        self.sent_tokens_offset = 0  # Offset of sent tokens\n",
                "        self.sent_tokens_offset = 0  # Offset of sent tokens\n"
                "        # AIGEN_UNCERTAINTY_PATCH:request-state\n"
                "        self.log_variances: list[float] = []\n"
                "        self.lm_head_variances: list[float] = []\n",
            ),
            Replacement(
                "            new_token_ids = engine_core_output.new_token_ids\n"
                "            pooling_output = engine_core_output.pooling_output\n",
                "            new_token_ids = engine_core_output.new_token_ids\n"
                "            # AIGEN_UNCERTAINTY_PATCH:accumulate\n"
                "            if engine_core_output.new_log_variances:\n"
                "                req_state.log_variances.extend(\n"
                "                    engine_core_output.new_log_variances\n"
                "                )\n"
                "            if engine_core_output.new_lm_head_variances:\n"
                "                req_state.lm_head_variances.extend(\n"
                "                    engine_core_output.new_lm_head_variances\n"
                "                )\n"
                "            pooling_output = engine_core_output.pooling_output\n",
            ),
            Replacement(
                "            finish_reason=str(finish_reason) if finished else None,\n"
                "            stop_reason=stop_reason if finished else None,\n"
                "        )\n",
                "            finish_reason=str(finish_reason) if finished else None,\n"
                "            stop_reason=stop_reason if finished else None,\n"
                "            # AIGEN_UNCERTAINTY_PATCH:completion-values\n"
                "            log_variances=(\n"
                "                list(self.log_variances)\n"
                "                if not delta\n"
                "                else list(self.log_variances[-len(token_ids) :])\n"
                "                if token_ids\n"
                "                else []\n"
                "            ),\n"
                "            lm_head_variances=(\n"
                "                list(self.lm_head_variances)\n"
                "                if not delta\n"
                "                else list(self.lm_head_variances[-len(token_ids) :])\n"
                "                if token_ids\n"
                "                else []\n"
                "            ),\n"
                "        )\n",
            ),
        ),
    ),
    FilePatch(
        "entrypoints/openai/chat_completion/protocol.py",
        (
            Replacement(
                "        extra_args: dict[str, Any] = self.vllm_xargs if self.vllm_xargs else {}\n",
                "        extra_args: dict[str, Any] = self.vllm_xargs if self.vllm_xargs else {}\n"
                "        # AIGEN_UNCERTAINTY_PATCH:lm-head-request-propagation\n"
                "        if self.return_lm_head_variance:\n"
                "            extra_args[\"return_lm_head_variance\"] = True\n",
            ),
            Replacement(
                "class ChatCompletionResponseChoice(OpenAIBaseModel):\n",
                "# AIGEN_UNCERTAINTY_PATCH:response-schema\n"
                "class UncertaintyInfo(OpenAIBaseModel):\n"
                "    variance: float\n"
                "    aggregation: Literal[\"mean\", \"last\"]\n"
                "    last_log_variance: float\n"
                "    mean_log_variance: float | None = None\n"
                "    token_variances: list[float] | None = None\n"
                "    token_log_variances: list[float] | None = None\n\n\n"
                "class LMHeadVarianceInfo(OpenAIBaseModel):\n"
                "    variance: float\n"
                "    aggregation: Literal[\"mean\", \"last\"]\n"
                "    last_variance: float\n"
                "    token_variances: list[float] | None = None\n"
                "    source: Literal[\"raw_lm_head_logits\"] = \"raw_lm_head_logits\"\n\n\n"
                "class ChatCompletionResponseChoice(OpenAIBaseModel):\n",
            ),
            Replacement(
                "    routed_experts: str | None = None\n\n\n"
                "class ChatCompletionResponse(OpenAIBaseModel):\n",
                "    routed_experts: str | None = None\n"
                "    uncertainty: UncertaintyInfo | None = None\n"
                "    lm_head_variance: LMHeadVarianceInfo | None = None\n\n\n"
                "class ChatCompletionResponse(OpenAIBaseModel):\n",
            ),
            Replacement(
                "    # not part of the OpenAI spec but for tracing the tokens\n"
                "    token_ids: list[int] | None = None\n\n\n"
                "class ChatCompletionStreamResponse(OpenAIBaseModel):\n",
                "    # not part of the OpenAI spec but for tracing the tokens\n"
                "    token_ids: list[int] | None = None\n"
                "    uncertainty: UncertaintyInfo | None = None\n"
                "    lm_head_variance: LMHeadVarianceInfo | None = None\n\n\n"
                "class ChatCompletionStreamResponse(OpenAIBaseModel):\n",
            ),
            Replacement(
                "    return_token_offsets: bool | None = Field(\n",
                "    # AIGEN_UNCERTAINTY_PATCH:request-option\n"
                "    return_token_variances: bool = Field(\n"
                "        default=False,\n"
                "        description=(\n"
                "            \"Return generated-token variance and log-variance lists. \"\n"
                "            \"The aggregate variance is returned regardless.\"\n"
                "        ),\n"
                "    )\n"
                "    return_lm_head_variance: bool = Field(\n"
                "        default=False,\n"
                "        description=(\n"
                "            \"Return population variance over raw LM-head vocabulary \"\n"
                "            \"logits. Non-streaming returns the generated-token mean; \"\n"
                "            \"streaming returns the current chunk's last-token value.\"\n"
                "        ),\n"
                "    )\n"
                "    return_token_lm_head_variances: bool = Field(\n"
                "        default=False,\n"
                "        description=(\n"
                "            \"Include the per-generated-token raw LM-head logit \"\n"
                "            \"variance list in a non-streaming response. Requires \"\n"
                "            \"return_lm_head_variance=true.\"\n"
                "        ),\n"
                "    )\n"
                "    return_token_offsets: bool | None = Field(\n",
            ),
        ),
    ),
    FilePatch(
        "entrypoints/openai/chat_completion/serving.py",
        (
            Replacement(
                "import asyncio\n"
                "import io\n"
                "import time\n",
                "import asyncio\n"
                "import io\n"
                "import math\n"
                "import os\n"
                "import time\n",
            ),
            Replacement(
                "    ChatCompletionStreamResponse,\n"
                "    ChatMessage,\n"
                ")\n",
                "    ChatCompletionStreamResponse,\n"
                "    ChatMessage,\n"
                "    LMHeadVarianceInfo,\n"
                "    UncertaintyInfo,\n"
                ")\n",
            ),
            Replacement(
                "logger = init_logger(__name__)\n\n\n"
                "def _get_mm_token_counts",
                "logger = init_logger(__name__)\n\n\n"
                "# AIGEN_UNCERTAINTY_PATCH:response-helper\n"
                "def _build_uncertainty_info(\n"
                "    log_variances: GenericSequence[float] | None,\n"
                "    aggregation: str,\n"
                "    include_tokens: bool,\n"
                ") -> UncertaintyInfo | None:\n"
                "    if not log_variances:\n"
                "        return None\n"
                "    log_values = [float(value) for value in log_variances]\n"
                "    variance_values = [math.exp(value) for value in log_values]\n"
                "    if aggregation == \"last\":\n"
                "        variance = variance_values[-1]\n"
                "        mean_log_variance = None\n"
                "    else:\n"
                "        variance = sum(variance_values) / len(variance_values)\n"
                "        mean_log_variance = sum(log_values) / len(log_values)\n"
                "    return UncertaintyInfo(\n"
                "        variance=variance,\n"
                "        aggregation=aggregation,\n"
                "        last_log_variance=log_values[-1],\n"
                "        mean_log_variance=mean_log_variance,\n"
                "        token_variances=variance_values if include_tokens else None,\n"
                "        token_log_variances=log_values if include_tokens else None,\n"
                "    )\n\n\n"
                "def _build_lm_head_variance_info(\n"
                "    variances: GenericSequence[float] | None,\n"
                "    aggregation: str,\n"
                "    include_tokens: bool,\n"
                ") -> LMHeadVarianceInfo | None:\n"
                "    if not variances:\n"
                "        return None\n"
                "    values = [float(value) for value in variances]\n"
                "    aggregate = (\n"
                "        values[-1]\n"
                "        if aggregation == \"last\"\n"
                "        else sum(values) / len(values)\n"
                "    )\n"
                "    return LMHeadVarianceInfo(\n"
                "        variance=aggregate,\n"
                "        aggregation=aggregation,\n"
                "        last_variance=values[-1],\n"
                "        token_variances=values if include_tokens else None,\n"
                "    )\n\n\n"
                "def _get_mm_token_counts",
            ),
            Replacement(
                "    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:\n"
                "        # Streaming response\n",
                "    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:\n"
                "        # AIGEN_UNCERTAINTY_PATCH:lm-head-request-validation\n"
                "        if (\n"
                "            request.return_token_lm_head_variances\n"
                "            and not request.return_lm_head_variance\n"
                "        ):\n"
                "            return self.create_error_response(\n"
                "                \"return_token_lm_head_variances requires \"\n"
                "                \"return_lm_head_variance=true\"\n"
                "            )\n"
                "        if (\n"
                "            request.return_lm_head_variance\n"
                "            and os.getenv(\"VLLM_ENABLE_LM_HEAD_VARIANCE\") != \"1\"\n"
                "        ):\n"
                "            return self.create_error_response(\n"
                "                \"LM-head variance is disabled; restart vllm serve with \"\n"
                "                \"--enable-lm-head-variance\"\n"
                "            )\n"
                "        # Streaming response\n",
            ),
            Replacement(
                "                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)\n"
                "                    chunk = ChatCompletionStreamResponse(\n",
                "                    # AIGEN_UNCERTAINTY_PATCH:stream-step\n"
                "                    choice_data.uncertainty = _build_uncertainty_info(\n"
                "                        output.log_variances,\n"
                "                        aggregation=\"last\",\n"
                "                        include_tokens=request.return_token_variances,\n"
                "                    )\n"
                "                    if request.return_lm_head_variance:\n"
                "                        choice_data.lm_head_variance = (\n"
                "                            _build_lm_head_variance_info(\n"
                "                                output.lm_head_variances,\n"
                "                                aggregation=\"last\",\n"
                "                                include_tokens=False,\n"
                "                            )\n"
                "                        )\n"
                "                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)\n"
                "                    chunk = ChatCompletionStreamResponse(\n",
            ),
            Replacement(
                "            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)\n\n"
                "            choices.append(choice_data)\n",
                "            choice_data.uncertainty = _build_uncertainty_info(\n"
                "                output.log_variances,\n"
                "                aggregation=\"mean\",\n"
                "                include_tokens=request.return_token_variances,\n"
                "            )\n"
                "            if request.return_lm_head_variance:\n"
                "                choice_data.lm_head_variance = (\n"
                "                    _build_lm_head_variance_info(\n"
                "                        output.lm_head_variances,\n"
                "                        aggregation=\"mean\",\n"
                "                        include_tokens=(\n"
                "                            request.return_token_lm_head_variances\n"
                "                        ),\n"
                "                    )\n"
                "                )\n"
                "            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)\n\n"
                "            choices.append(choice_data)\n",
            ),
        ),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_vllm(target: str | None) -> Path:
    if target:
        candidate = Path(target).expanduser().resolve()
        if (candidate / "vllm" / "__init__.py").is_file():
            candidate = candidate / "vllm"
        if not (candidate / "__init__.py").is_file():
            raise PatchError(
                f"--target must be a vLLM package or repo root: {candidate}"
            )
        return candidate

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise PatchError("vllm is not installed in this Python environment")
    return Path(spec.origin).resolve().parent


def installed_version() -> str | None:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None


def transform(file_patch: FilePatch, source: str) -> str:
    result = source
    for replacement in file_patch.replacements:
        count = result.count(replacement.old)
        if count != replacement.expected:
            raise PatchError(
                f"{file_patch.relative_path}: expected anchor {replacement.expected} "
                f"time(s), found {count}"
            )
        result = result.replace(replacement.old, replacement.new)
    return result


def patch_state(package_root: Path) -> str:
    states: list[str] = []
    for file_patch in PATCHES:
        path = package_root / file_patch.relative_path
        if not path.is_file():
            states.append("missing")
            continue
        text = path.read_text(encoding="utf-8")
        states.append("patched" if "AIGEN_UNCERTAINTY_PATCH" in text else "original")
    unique = set(states)
    if unique == {"original"}:
        return "original"
    if unique == {"patched"}:
        return "patched"
    return "partial:" + ",".join(states)


def validate_original(package_root: Path, force: bool) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for file_patch in PATCHES:
        path = package_root / file_patch.relative_path
        if not path.is_file():
            raise PatchError(f"required vLLM file not found: {path}")
        actual = sha256(path)
        hashes[file_patch.relative_path] = actual
        expected = EXPECTED_SHA256[file_patch.relative_path]
        if actual != expected and not force:
            raise PatchError(
                f"unsupported {file_patch.relative_path} hash: {actual}\n"
                f"expected vLLM {SUPPORTED_VLLM_VERSION}: {expected}\n"
                "Use --force only after reviewing the source anchors."
            )
        transform(file_patch, path.read_text(encoding="utf-8"))
    return hashes


def atomic_write(path: Path, content: str) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify(package_root: Path) -> None:
    if patch_state(package_root) != "patched":
        raise PatchError("patch is not fully applied")
    for file_patch in PATCHES:
        py_compile.compile(str(package_root / file_patch.relative_path), doraise=True)
    print(f"Verified {len(PATCHES)} patched Python files.")


def apply_patch(package_root: Path, force: bool, yes: bool) -> None:
    state = patch_state(package_root)
    if state == "patched":
        verify(package_root)
        print(f"Already patched: {package_root}")
        return
    if state != "original":
        raise PatchError(f"refusing to patch inconsistent tree ({state})")

    version = installed_version()
    if version is not None and version != SUPPORTED_VLLM_VERSION and not force:
        raise PatchError(
            f"installed vLLM is {version}; this patch supports "
            f"{SUPPORTED_VLLM_VERSION} only"
        )
    original_hashes = validate_original(package_root, force)
    transformed = {
        file_patch.relative_path: transform(
            file_patch,
            (package_root / file_patch.relative_path).read_text(encoding="utf-8"),
        )
        for file_patch in PATCHES
    }

    print(f"Target: {package_root}")
    print(f"vLLM:  {version or 'source checkout'}")
    print(f"Files: {len(transformed)}")
    if not yes:
        answer = input("Apply uncertainty/LM-head patch? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled")
            return

    backup_root = package_root / BACKUP_DIRNAME
    if backup_root.exists():
        raise PatchError(f"backup directory already exists: {backup_root}")
    backup_root.mkdir(parents=True)
    manifest = {
        "patch_id": PATCH_ID,
        "vllm_version": version,
        "package_root": str(package_root),
        "original_sha256": original_hashes,
        "files": list(transformed),
    }
    try:
        for relative_path in transformed:
            source = package_root / relative_path
            destination = backup_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (backup_root / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        for relative_path, content in transformed.items():
            atomic_write(package_root / relative_path, content)
    except Exception:
        for relative_path in transformed:
            backup = backup_root / relative_path
            if backup.is_file():
                shutil.copy2(backup, package_root / relative_path)
        raise

    verify(package_root)
    print("Patch applied successfully.")
    print("Restart every vLLM process before serving requests.")


def revert_patch(package_root: Path, yes: bool) -> None:
    backup_root = package_root / BACKUP_DIRNAME
    manifest_path = backup_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PatchError(f"backup manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_patch_id = manifest.get("patch_id")
    if manifest_patch_id != PATCH_ID and manifest_patch_id not in LEGACY_PATCH_IDS:
        raise PatchError("backup belongs to a different patch")
    files = manifest["files"]
    print(f"Restore {len(files)} files in {package_root}")
    if not yes:
        answer = input("Revert uncertainty patch? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled")
            return
    for relative_path in files:
        backup = backup_root / relative_path
        if not backup.is_file():
            raise PatchError(f"backup file missing: {backup}")
        shutil.copy2(backup, package_root / relative_path)
    shutil.rmtree(backup_root)
    if patch_state(package_root) != "original":
        raise PatchError("revert completed but original-state check failed")
    print("Patch reverted successfully.")


def show_status(package_root: Path) -> None:
    version = installed_version()
    print(f"package_root={package_root}")
    print(f"installed_version={version or 'unknown/source-checkout'}")
    print(f"supported_version={SUPPORTED_VLLM_VERSION}")
    print(f"state={patch_state(package_root)}")
    print(f"backup={(package_root / BACKUP_DIRNAME).is_dir()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch vLLM 0.26.0 with Ministral3 uncertainty and raw "
            "LM-head logit variance output"
        )
    )
    parser.add_argument("command", choices=("status", "check", "apply", "revert"))
    parser.add_argument(
        "--target",
        help="vLLM package directory or source repo root; defaults to installed vLLM",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow hash/version mismatch when exact source anchors still match",
    )
    parser.add_argument(
        "--yes", action="store_true", help="do not prompt before apply/revert"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        package_root = locate_vllm(args.target)
        if args.command == "status":
            show_status(package_root)
        elif args.command == "check":
            state = patch_state(package_root)
            if state == "patched":
                verify(package_root)
            elif state == "original":
                validate_original(package_root, args.force)
                print("Compatible original vLLM tree; ready to apply.")
            else:
                raise PatchError(f"inconsistent tree: {state}")
        elif args.command == "apply":
            apply_patch(package_root, args.force, args.yes)
        elif args.command == "revert":
            revert_patch(package_root, args.yes)
        return 0
    except (PatchError, OSError, py_compile.PyCompileError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
