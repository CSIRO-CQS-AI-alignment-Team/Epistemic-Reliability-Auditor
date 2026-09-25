"""Shared debate engine and standalone transcript-generation CLI.

``debate-bok.py`` and ``adversarial_transcript.generate`` reuse the model clients,
``Debater``, ``debate()``, and ``stance_rng()`` defined here.
"""

import argparse
import glob
import json
import os
import random
import re
import tempfile

import yaml
from tqdm import tqdm

# The stdlib-only quote helpers are shared with judge and verifier code without
# importing this module's model clients or YAML configuration.
from quote_utils import normalize_text, verify_quotes


# ---- Configuration ----
NUM_ROUNDS = 3  # rollout/sim.yaml: num_steps (overridable via --num-rounds)
CONFIG_PATH = "config/debate-default.yaml"
SEPARATOR = "---------------------------------------------\n\n"  # transcript turn separator
SEED = 1090998506967575423  # fixed stance seed (from dataset/metadata.json random_seed)

# Backend defaults for the standalone CLI and importing generators.
DEFAULT_BACKEND = "transformers"  # "api" or "transformers"
DEFAULT_MODEL = "google/gemma-4-31B-it"  # local vLLM model
BASE_URL = "http://localhost:18888/v1"  # hardcoded vLLM endpoint (vLLM ignores the key)
DEFAULT_API_KEY = "EMPTY"

# Default data paths (relative to the working directory; run from the repo root).
DEFAULT_INPUT = "dataset/QuALITY-H.json"
DEFAULT_STORY_MAP = "dataset/title-story.json"
DEFAULT_OUTPUT = "dataset/QuALITY-H-with-honest-transcripts.json"
EXPECTED_EXAMPLES = 222  # default full-run validation size (QuALITY-H)
SAVE_EVERY = 5  # progress save: flush every N freshly generated transcripts
# torchrun: gloo barrier timeout. A fast/empty rank waits at the end barrier for the
# SLOWEST rank's FULL runtime, so set this above the longest expected total run time.
DDP_TIMEOUT_HOURS = 24

def load_config(path):
    """Load one debate config without mutating the module-level default.

    `debate-bok.py` passes an explicitly loaded config through ``debate(..., config=)``
    for transcript-length ablations.  Ordinary debate/adversarial callers keep using
    ``CONFIG`` below, so one honest-BoK run cannot leak settings through globals.
    """
    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: debate config must be a YAML mapping")
    return config


CONFIG = load_config(CONFIG_PATH)
PARTIALS = CONFIG["prompts"]["partials"]
LM_CONFIG = CONFIG["language_model"]
MESSAGES = CONFIG["prompts"]["messages"]  # static across rounds; read once
WORD_LIMIT = CONFIG["prompts"].get("word_limit", 100)  # prompt-stated per-argument limit


def extract_argument(response):
    """Keep only <argument>...</argument> (drop the private <thinking>), like the reference.

    The judge and the opposing debater must never see the <thinking> block, so it is
    stripped before the response is written into the transcript.
    """
    # The model must not mark its own quotes verified; collapse any v/u tags it
    # emitted back to plain <quote> so verification is decided only against the story.
    for tag in ("<v_quote>", "<u_quote>"):
        response = response.replace(tag, "<quote>")
    for tag in ("</v_quote>", "</u_quote>"):
        response = response.replace(tag, "</quote>")
    if "<argument>" in response and "</argument>" in response:
        return response.split("<argument>", 1)[1].split("</argument>", 1)[0].strip()
    # Fallback: no well-formed <argument> tag. Strip any <thinking>...</thinking>
    # so the debater's private reasoning still never leaks into the transcript.
    return re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL).strip()


def truncate(argument, max_words, marker_within_limit=False):
    """Truncate an argument under one of two explicit whitespace-word policies.

    When ``marker_within_limit`` is false, ``max_words`` applies to content and the
    truncation marker is appended afterward. When true, the configured bound covers all
    stored whitespace words, including the marker.
    """
    if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words < 1:
        raise ValueError("max_words must be a positive integer")
    if not marker_within_limit:
        # Content-cap policy: split(" ") retains empty fields between repeated spaces
        # and leaves newlines inside fields. The marker is outside the content cap.
        words = argument.split(" ")
        truncated = len(words) > max_words
        new_arg = " ".join(words[:max_words])
        marker = "... <TRUNCATED>" if truncated else ""
    else:
        # Strict ablation path.  Count the same whitespace words as
        # debate-bok.assert_transcript_budget, but slice the ORIGINAL string so paragraph
        # breaks and interior spacing are not a hidden formatting intervention.
        matches = list(re.finditer(r"\S+", argument))
        if len(matches) <= max_words:
            new_arg = argument
            marker = ""
        else:
            marker_words = 2 if max_words >= 2 else 1
            content_budget = max_words - marker_words
            new_arg = argument[:matches[content_budget - 1].end()] if content_budget else ""
            marker = "... <TRUNCATED>" if marker_words == 2 else "<TRUNCATED>"
    # Don't leave a quote tag dangling open (mirrors the reference truncate).
    for tag in ("quote", "u_quote", "v_quote"):
        if f"<{tag}>" in new_arg and f"</{tag}>" not in new_arg.split(f"<{tag}>")[-1]:
            new_arg += f"</{tag}>"
    if marker:
        separator = " " if marker_within_limit and new_arg else ""
        return new_arg + separator + marker
    return new_arg


def render_transcript(completed_rounds, our_name, opponent_name, story_normalised):
    """Egocentric debate transcript (mirrors debater_quality.get_transcript_string).

    This debater's own argument is listed first each round, rounds are joined by
    SEPARATOR, and a bare "Round N:" header is appended for the round currently being
    generated (the empty round quality_sim pre-appends before each turn). Quotes are
    re-tagged verified/unverified against the story, as the reference does on render.
    """
    rounds_for_render = completed_rounds + [{}]  # + current (empty) round
    n = len(rounds_for_render)
    out = ""
    for i, rnd in enumerate(rounds_for_render):
        out += f"Round {i + 1}:\n\n"
        our_arg = rnd.get(our_name)
        opponent_arg = rnd.get(opponent_name)
        if our_arg is not None:
            out += f'{our_name}: """{verify_quotes(our_arg, story_normalised)}"""\n\n'
        if opponent_arg is not None:
            out += f'{opponent_name}: """{verify_quotes(opponent_arg, story_normalised)}"""\n\n'
        if i + 1 < n:
            out += f"{SEPARATOR}\n\n"
    return out.strip()


# ---- Model backends ----
class BaseModelClient:
    """Uniform chat interface: generate(messages) -> assistant text."""

    def generate(self, messages):
        raise NotImplementedError


class ApiModelClient(BaseModelClient):
    """OpenAI-compatible / vLLM backend used by the standalone debate CLI."""

    def __init__(self, base_url, api_key, model, lm_config):
        from openai import OpenAI  # lazy: the API path must not require torch/transformers

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.lm = lm_config

    def generate(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.lm["temperature"],
            top_p=self.lm["top_p"],
            max_tokens=self.lm["max_tokens"],
            timeout=self.lm["timeout"],
        )
        # Length/filter/tool finishes can yield None content; never return None.
        return completion.choices[0].message.content or ""


class TransformersModelClient(BaseModelClient):
    """Local HuggingFace backend; model + tokenizer are loaded exactly once."""

    def __init__(self, model, lm_config, torch_dtype="auto", device_map="auto", local_rank=None):
        import torch  # lazy: only the transformers path needs the GPU stack
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.lm = lm_config
        if local_rank is not None:
            # DDP: pin this process to its own card before loading the replica.
            n_gpu = torch.cuda.device_count()
            if local_rank >= n_gpu:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only {n_gpu} CUDA device(s) visible — "
                    "--nproc_per_node exceeds the GPUs on this node."
                )
            torch.cuda.set_device(local_rank)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = "auto" if torch_dtype in (None, "auto") else getattr(torch, torch_dtype)
        try:
            # `dtype=` matches the repo's judge-oss.py (recent transformers).
            self.model = AutoModelForCausalLM.from_pretrained(
                model, dtype=dtype, device_map=device_map
            )
        except TypeError:
            # Fallback for Transformers versions that accept ``torch_dtype`` instead of ``dtype``.
            self.model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=dtype, device_map=device_map
            )
        self.model.eval()

    def _render_prompt(self, messages):
        """Build the prompt string. Returns (prompt, add_special_tokens)."""
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # The template already injected BOS/special tokens; don't add them again.
            return prompt, False
        # Fallback for tokenizers without a chat template: plain role/content concat.
        prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return prompt + "\n\nassistant:", True

    def generate(self, messages):
        torch = self._torch
        prompt, add_special_tokens = self._render_prompt(messages)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=add_special_tokens
        ).to(self.model.device)

        temperature = self.lm["temperature"]
        do_sample = bool(temperature and temperature > 0)
        gen_kwargs = {"max_new_tokens": self.lm["max_tokens"]}
        if self.tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=self.lm["top_p"])
        else:
            gen_kwargs["do_sample"] = False

        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = outputs[0][prompt_len:]  # return ONLY the newly generated text
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_model_client(args, device_map=None, local_rank=None):
    if args.backend == "api":
        return ApiModelClient(BASE_URL, args.api_key, args.model, LM_CONFIG)
    if args.backend == "transformers":
        return TransformersModelClient(
            args.model,
            LM_CONFIG,
            torch_dtype=args.torch_dtype,
            device_map=device_map if device_map is not None else args.device_map,
            local_rank=local_rank,
        )
    raise ValueError(f"unknown backend: {args.backend}")


class Debater:
    def __init__(
        self, name, opponent_name, question, defending_answer, opposing_answer, story,
        model_client, config=None, marker_within_limit=False
    ):
        self.name = name
        self.opponent_name = opponent_name
        self.question = question
        self.defending_answer = defending_answer
        self.opposing_answer = opposing_answer
        self.article = story
        self.story_normalised = normalize_text(story)  # for quote verification
        self.model_client = model_client
        self.marker_within_limit = bool(marker_within_limit)
        self.config = CONFIG if config is None else config
        self.partials = self.config["prompts"]["partials"]
        self.lm_config = self.config["language_model"]
        self.word_limit = self.config["prompts"].get("word_limit", 100)

        messages = self.config["prompts"]["messages"]
        # Messages 0-2 are static across rounds; fill them once.
        self.system = {
            "role": messages[0]["role"],
            "content": messages[0]["content"]
            .replace("<NAME>", name)
            .replace("<WORD_LIMIT>", str(self.word_limit)),
        }
        self.user = {
            "role": messages[1]["role"],
            "content": messages[1]["content"]
            .replace("<FEW_SHOT_MESSAGE>", "")
            .replace("<QUESTION>", question)
            .replace("<ANSWER_DEFENDING>", defending_answer)
            .replace("<ANSWER_OPPOSING>", opposing_answer),
        }
        self.assistant = {
            "role": messages[2]["role"],
            "content": messages[2]["content"]
            .replace("<QUESTION>", question)
            .replace("<ANSWER_DEFENDING>", defending_answer)
            .replace("<ANSWER_OPPOSING>", opposing_answer),
        }
        # Message 3 is the per-round turn template; kept raw and re-filled each round.
        self.turn_role = messages[3]["role"]
        self.turn_template = messages[3]["content"]

    def build_turn_message(self, completed_rounds):
        # Partials are selected by how many arguments THIS debater has already made:
        #    0  -> opening + first_round_thinking    (round 1)
        #    1  -> next    + second_round_thinking   (round 2)
        #   >=2 -> next    + nth_round_thinking      (round 3+)
        our_args = len(completed_rounds)
        if our_args == 0:
            arg_request = self.partials["opening_argument_request"]
            thinking = self.partials["first_round_thinking"]
        else:
            arg_request = self.partials["nth_argument_request"]
            thinking = (
                self.partials["second_round_thinking"]
                if our_args == 1
                else self.partials["nth_round_thinking"]
            )

        transcript_str = render_transcript(
            completed_rounds, self.name, self.opponent_name, self.story_normalised
        )
        # Order matters: insert the partials (which themselves contain <TRANSCRIPT>,
        # <QUESTION>, <ANSWER_DEFENDING>) before filling those inner placeholders.
        content = (
            self.turn_template
            .replace("<STORY>", self.article)
            .replace("<TRANSCRIPT_MESSAGE>", self.partials["transcript"])
            .replace("<TRANSCRIPT>", transcript_str)
            .replace("<NEW_ARGUMENT_REQUEST>", arg_request)
            .replace("<THINKING_ADVICE>", thinking)
            .replace("<QUESTION>", self.question)
            .replace("<ANSWER_DEFENDING>", self.defending_answer)
            .replace("<WORD_LIMIT>", str(self.word_limit))
        )
        return {"role": self.turn_role, "content": content}

    def take_turn(self, completed_rounds):
        messages = [
            self.system,
            self.user,
            self.assistant,
            self.build_turn_message(completed_rounds),
        ]
        response = self.query(messages)
        return truncate(
            extract_argument(response), self.lm_config["max_words"],
            marker_within_limit=self.marker_within_limit)

    def query(self, messages):
        return self.model_client.generate(messages)


def debate(question, y_true, y_false, story, model_client, rng, num_rounds=NUM_ROUNDS,
           config=None, marker_within_limit=False):
    """Run one honest debate over q_m and return its transcript dict.

    `rng` (a seeded random.Random) decides the stance 50/50: Debater A defends either
    Y_true or Y_false, and Debater B defends the other. The returned dict records each
    debater's stance label plus the per-round arguments. q_h is never referenced here.
    """
    a_defends_true = rng.random() < 0.5
    if a_defends_true:
        a_def, a_opp, a_label, b_label = y_true, y_false, "Y_true", "Y_false"
    else:
        a_def, a_opp, a_label, b_label = y_false, y_true, "Y_false", "Y_true"

    debaterA = Debater("Debater A", "Debater B", question, a_def, a_opp, story,
                       model_client, config=config,
                       marker_within_limit=marker_within_limit)
    debaterB = Debater("Debater B", "Debater A", question, a_opp, a_def, story,
                       model_client, config=config,
                       marker_within_limit=marker_within_limit)

    rounds = []
    for _ in range(num_rounds):
        # Simultaneous round: both debaters answer on the SAME prior transcript,
        # so neither sees the other's current-round argument.
        arg_a = debaterA.take_turn(rounds)
        arg_b = debaterB.take_turn(rounds)
        rounds.append({"Debater A": arg_a, "Debater B": arg_b})

    return {"Debater A": a_label, "Debater B": b_label, "rounds": rounds}


# ---- Driver ----
def stance_rng(example):
    """Per-example deterministic RNG seeded from the fixed SEED.

    Keyed on stable content (title, question, both answers) so each q_m's stance is
    identical regardless of processing order or which examples were skipped/resumed.
    random.Random(str) seeds via SHA-512, so this is reproducible across runs.
    """
    q_y = example["Q_Y"]
    key = "|".join(
        str(part)
        for part in (SEED, example.get("story_title"), q_y["question"], q_y["Y_true"], q_y["Y_false"])
    )
    return random.Random(key)


def atomic_write_json(path, data):
    """Write JSON to a temp file in the destination dir, then os.replace (atomic)."""
    dest_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=dest_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ---- DDP (torchrun) per-rank progress files ----
# Under `torchrun --standalone --nproc_per_node=N` each rank writes its own shard's
# progress to `{save_path}.rank{r}.json`. torchrun only spawns the processes + sets
# env (LOCAL_RANK/RANK/WORLD_SIZE); a single gloo `dist.barrier()` synchronises the
# end of the run, after which rank 0 merges the rank files into one output.
def run_signature(args):
    """Identifies a run for safe rank-file reuse. Transcripts depend on these, so a
    rank file whose signature differs from the current run is treated as stale."""
    return {
        "model": args.model,
        "num_rounds": args.num_rounds,
        "seed": SEED,
        "input": os.path.realpath(args.input),
        "story_map": os.path.realpath(args.story_map),
    }


def _rank_file(save_path, rank):
    return f"{save_path}.rank{rank}.json"


def write_rank_file(save_path, rank, sig, transcripts):
    """Atomically persist this rank's shard: {meta, transcripts: {global_idx: transcript}}."""
    payload = {"meta": sig, "transcripts": {str(gi): t for gi, t in transcripts.items()}}
    atomic_write_json(_rank_file(save_path, rank), payload)


def load_rank_results(save_path, sig, data_len):
    """Merge {global_idx: transcript} from every rank file whose meta matches `sig`.

    Stale / corrupt / mismatched-signature files are skipped with a warning, and each
    index is bounds-checked against data_len so a bad file can never write a transcript
    onto the wrong example.
    """
    results = {}
    for path in sorted(glob.glob(glob.escape(save_path) + ".rank*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[WARN] skipping unreadable rank file {path}: {type(e).__name__}: {e}")
            continue
        if not isinstance(payload, dict) or payload.get("meta") != sig:
            print(f"[WARN] ignoring rank file {path} (missing/mismatched run signature)")
            continue
        transcripts = payload.get("transcripts")
        if not isinstance(transcripts, dict):
            print(f"[WARN] rank file {path} has a malformed transcripts map; skipping")
            continue
        for k, t in transcripts.items():
            try:
                gi = int(k)
            except (TypeError, ValueError):
                print(f"[WARN] non-integer index {k!r} in {path}; skipping")
                continue
            if 0 <= gi < data_len:
                results[gi] = t
            else:
                print(f"[WARN] out-of-range index {gi} in {path}; skipping")
    return results


def cleanup_rank_files(save_path):
    for path in glob.glob(glob.escape(save_path) + ".rank*.json"):
        try:
            os.remove(path)
        except OSError as e:
            print(f"[WARN] could not remove rank file {path}: {e}")


def preflight(data, story_map, announce=True):
    """Cheap data sanity checks before loading a (possibly large) model."""
    if not isinstance(data, list):
        raise SystemExit(f"input must be a top-level JSON list, got {type(data).__name__}")
    if not isinstance(story_map, dict):
        raise SystemExit(
            f"story map must be a JSON object {{title: story}}, got {type(story_map).__name__}"
        )
    missing_keys = 0
    missing_story = 0
    for ex in data:
        q_y = ex.get("Q_Y")
        if not isinstance(q_y, dict) or not {"question", "Y_true", "Y_false"} <= set(q_y):
            missing_keys += 1
        if ex.get("story_title") not in story_map:
            missing_story += 1
    if missing_keys:
        raise SystemExit(
            f"{missing_keys} example(s) missing required Q_Y keys (question/Y_true/Y_false)"
        )
    if missing_story and announce:
        print(
            f"[WARN] {missing_story}/{len(data)} story_title(s) absent from the story map; "
            "those examples will be skipped"
        )


def validate(data, num_rounds, expected):
    """Return (errors, missing_count). errors is a list of structural-violation strings."""
    errors = []
    if expected is not None and len(data) != expected:
        errors.append(f"example count {len(data)} != {expected}")
    missing = 0
    for ex in data:
        title = ex.get("story_title")
        if "transcript" in ex:
            errors.append(f"'{title}': transcript found at top level (must live under Q_Y)")
        q_h = ex.get("Q_H")
        if isinstance(q_h, dict) and "transcript" in q_h:
            errors.append(f"'{title}': Q_H must not carry a transcript")
        q_y = ex.get("Q_Y") if isinstance(ex.get("Q_Y"), dict) else {}
        if "transcripts" in q_y:
            errors.append(f"'{title}': unexpected Q_Y['transcripts'] (plural)")
        t = q_y.get("transcript")
        if t is None:
            missing += 1
            continue
        if not isinstance(t, dict) or set(t.keys()) != {"Debater A", "Debater B", "rounds"}:
            errors.append(f"'{title}': transcript keys must be exactly {{Debater A, Debater B, rounds}}")
            continue
        if {t["Debater A"], t["Debater B"]} != {"Y_true", "Y_false"}:
            errors.append(f"'{title}': stance labels must be {{Y_true, Y_false}}")
        rounds = t["rounds"]
        if not isinstance(rounds, list) or len(rounds) != num_rounds:
            errors.append(f"'{title}': rounds length != {num_rounds}")
        else:
            for i, rnd in enumerate(rounds):
                if not isinstance(rnd, dict) or set(rnd.keys()) != {"Debater A", "Debater B"}:
                    errors.append(f"'{title}': round {i + 1} keys must be exactly {{Debater A, Debater B}}")
    return errors, missing


def parse_args():
    def positive_int(value):
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError("must be a positive integer (>= 1)")
        return ivalue

    p = argparse.ArgumentParser(
        description="Generate one honest debate transcript per QuALITY-H Q_Y.",
        epilog=(
            "Multi-GPU (transformers, data-parallel — one replica per card):\n"
            "  torchrun --standalone --nproc_per_node=4 debate.py --backend transformers --model <m>\n"
            "Single GPU / process:\n"
            "  CUDA_VISIBLE_DEVICES=0 python3 debate.py --backend transformers --model <m>\n"
            "Under torchrun, --device-map is overridden to this rank's card."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="QuALITY-H json (top-level list)")
    p.add_argument("--story-map", default=DEFAULT_STORY_MAP, help="{title: story} json")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="output json (input untouched)")
    p.add_argument("--in-place", action="store_true", help="write back to --input")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing transcripts")
    p.add_argument("--num-rounds", type=positive_int, default=NUM_ROUNDS)
    p.add_argument("--limit", type=positive_int, default=None, help="process only first N")
    p.add_argument(
        "--expected-examples",
        type=positive_int,
        default=EXPECTED_EXAMPLES,
        help=(
            "expected input size for full-run validation "
            f"(default: {EXPECTED_EXAMPLES}; e.g. 101 for GPQA)"
        ),
    )
    p.add_argument("--backend", choices=["api", "transformers"], default=DEFAULT_BACKEND)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY, help="api backend only")
    p.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="transformers backend only",
    )
    p.add_argument("--device-map", default="auto", help="transformers backend only")
    return p.parse_args()


def main():
    args = parse_args()

    # torchrun supplies these variables; their absence selects single-process execution.
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    distributed = local_rank >= 0
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if distributed else 1
    global_rank = int(os.environ.get("RANK", "0")) if distributed else 0
    is_main = (not distributed) or global_rank == 0

    if not args.in_place and os.path.realpath(args.output) == os.path.realpath(args.input):
        raise SystemExit(
            "--output equals --input; pass --in-place to overwrite the input deliberately."
        )
    save_path = args.input if args.in_place else args.output
    sig = run_signature(args)

    # Bring up the gloo (CPU) process group early so the end-of-run barrier is ready.
    # Used ONLY for synchronisation; model compute stays independent per rank/card.
    if distributed:
        import datetime
        import torch.distributed as dist

        # After this point an unhandled error on one rank (e.g. model OOM) makes it skip
        # the end barriers; torchrun detects the failed worker and tears the job down, and
        # progress stays in the per-rank files for the next run to resume from.
        dist.init_process_group(
            backend="gloo", timeout=datetime.timedelta(hours=DDP_TIMEOUT_HOURS)
        )
        if is_main and args.overwrite:
            cleanup_rank_files(save_path)  # wipe stale shards before regenerating
        dist.barrier()  # ranks wait for the (overwrite) wipe before any of them write

    # Resume-aware load: keep prior transcripts unless overwriting.
    load_path = save_path if (not args.overwrite and os.path.exists(save_path)) else args.input
    with open(load_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(args.story_map, encoding="utf-8") as f:
        story_map = json.load(f)

    preflight(data, story_map, announce=is_main)

    # Resume overlay (distributed): fold any prior per-rank progress into `data` so every
    # rank skips already-finished examples. Single-process resumes from save_path directly.
    if distributed and not args.overwrite:
        for gi, t in load_rank_results(save_path, sig, len(data)).items():
            data[gi]["Q_Y"]["transcript"] = t

    examples = data if args.limit is None else data[: args.limit]
    my_indices = (
        list(range(global_rank, len(examples), world_size))
        if distributed
        else list(range(len(examples)))
    )

    def needs_generation(gi):
        q_y = examples[gi]["Q_Y"]
        if q_y.get("transcript") is not None and not args.overwrite:
            return False
        return story_map.get(examples[gi].get("story_title")) is not None

    # Don't pay the (possibly multi-minute) model load when this rank has no work.
    if any(needs_generation(gi) for gi in my_indices):
        device_map = {"": local_rank} if (distributed and args.backend == "transformers") else None
        model_client = build_model_client(
            args, device_map=device_map, local_rank=local_rank if distributed else None
        )
    else:
        model_client = None

    def my_shard_transcripts():
        # Full done-state of THIS rank's shard (resume overlay + freshly generated).
        return {
            gi: examples[gi]["Q_Y"]["transcript"]
            for gi in my_indices
            if examples[gi]["Q_Y"].get("transcript") is not None
        }

    processed = skipped = failed = 0
    first_attempt = True
    desc = f"Q_Y debates (rank {global_rank})" if distributed else "Q_Y debates"
    for gi in tqdm(my_indices, desc=desc, disable=(distributed and not is_main)):
        q_y = examples[gi]["Q_Y"]
        if q_y.get("transcript") is not None and not args.overwrite:
            skipped += 1
            continue
        title = examples[gi].get("story_title")
        story = story_map.get(title)
        if story is None:
            failed += 1
            tqdm.write(f"[WARN] no story for '{title}'; skipping")
            continue
        rng = stance_rng(examples[gi])
        generated = False
        try:
            q_y["transcript"] = debate(
                q_y["question"],
                q_y["Y_true"],
                q_y["Y_false"],
                story,
                model_client,
                rng,
                num_rounds=args.num_rounds,
            )
            processed += 1
            generated = True
        except Exception as e:
            failed += 1
            msg = f"{type(e).__name__}: {e}"
            # Single-process aborts on a first-attempt failure (systematic bug/config).
            # Distributed tolerates per-example failures so the rank still reaches the
            # barrier (a mid-loop raise would hang the collective).
            if first_attempt and not distributed:
                raise RuntimeError(
                    f"first debate attempt failed ({msg}); aborting before wasting a long run"
                ) from e
            tqdm.write(f"[WARN] debate failed for '{title}': {msg}")
        finally:
            first_attempt = False
        # Progress save every SAVE_EVERY freshly generated transcripts.
        if generated and processed % SAVE_EVERY == 0:
            if distributed:
                write_rank_file(save_path, global_rank, sig, my_shard_transcripts())
            else:
                atomic_write_json(save_path, data)

    # ---- Single-process finalization ----------------------------------------
    if not distributed:
        atomic_write_json(save_path, data)
        print(f"processed={processed} skipped(reused)={skipped} failed={failed} -> {save_path}")
        if skipped:
            print(
                f"[NOTE] {skipped} existing transcript(s) reused as-is; skip ignores "
                "SEED/--model/--num-rounds changes. Pass --overwrite to regenerate."
            )
        full_run = args.limit is None
        errors, missing = validate(
            data, args.num_rounds, args.expected_examples if full_run else None
        )
        if errors or (full_run and missing):
            for err in errors:
                print(f"[VALIDATION] {err}")
            if full_run and missing:
                print(f"[VALIDATION] {missing} Q_Y example(s) missing a transcript")
            raise SystemExit("validation FAILED")
        note = f" ({missing} still missing; rerun to fill)" if missing else ""
        print(f"validation PASSED: {len(data)} examples, transcripts well-formed{note}")
        return

    # ---- Distributed: final per-rank write, barrier, then rank 0 merges to one file ----
    write_rank_file(save_path, global_rank, sig, my_shard_transcripts())
    dist.barrier()  # every rank has finished writing its (final) rank file

    status_fail = False
    if is_main:
        try:
            for gi, t in load_rank_results(save_path, sig, len(data)).items():
                data[gi]["Q_Y"]["transcript"] = t
            atomic_write_json(save_path, data)  # WRITE before validate (never lose merged work)
            present = sum(
                1
                for ex in data
                if isinstance(ex.get("Q_Y"), dict) and ex["Q_Y"].get("transcript") is not None
            )
            print(f"[merge] {present}/{len(data)} transcripts present -> {save_path}")
            full_run = args.limit is None
            errors, missing = validate(
                data, args.num_rounds, args.expected_examples if full_run else None
            )
            if errors or (full_run and missing):
                for err in errors:
                    print(f"[VALIDATION] {err}")
                if full_run and missing:
                    print(f"[VALIDATION] {missing} Q_Y example(s) missing a transcript")
                status_fail = True
            else:
                cleanup_rank_files(save_path)  # only on full success; else keep for resume
                note = f" ({missing} still missing; rerun to fill)" if missing else ""
                print(f"validation PASSED: {len(data)} examples, transcripts well-formed{note}")
        except BaseException as e:
            print(f"[ERROR] rank-0 finalize failed: {type(e).__name__}: {e}")
            status_fail = True

    dist.barrier()  # keep every rank in the group until rank 0 has finished merging
    dist.destroy_process_group()
    if is_main and status_fail:
        raise SystemExit("validation FAILED")


if __name__ == "__main__":
    main()
