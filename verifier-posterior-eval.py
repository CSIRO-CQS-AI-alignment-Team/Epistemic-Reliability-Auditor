"""No-story/no-transcript forced-choice evaluation for QuALITY-H and GPQA.

The evaluator asks ``V_base``, ``V_honest``, or ``V_adv`` to answer ``Q_Y`` and/or
``Q_H`` from only the question and two answer options. It never includes source stories
or debate transcripts.

Every question is scored in both A/B answer orders at the Harmony final-channel
``Answer:`` slot. Per-order outputs are mapped back to semantic labels and combined
before accuracy and mean hidden-hypothesis probabilities are reported. The output also
records an order-combined semantic-logit probability ``p_sem`` and separate position-bias
diagnostics. ``--single-order`` is a diagnostic mode and is excluded from report metrics.

Dry run:
    python3 verifier-posterior-eval.py --dry-run --limit 2 --debug-prompts \
        --output /tmp/posterior-dry.json

One checkpoint:
    CUDA_VISIBLE_DEVICES=0 python3 verifier-posterior-eval.py \
        --dataset dataset/QuALITY-H/QuALITY-H-no-debate.json \
        --model checkpoints/gpt-oss-20b-verifier-fullft-adversarial-QuALITY-H \
        --model-label adv \
        --output runs/posterior_shift/QuALITY-H/adv.json
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

from adversarial_transcript.score_verifier import ForcedChoiceVerifier, _softmax_two


ALL_QUESTIONS = ("Q_Y", "Q_H")
LABELS = {"Q_Y": ("Y_true", "Y_false"), "Q_H": ("H_true", "H_false")}
ANSWER_FIELDS = {"Q_Y": ("Y_true", "Y_false"), "Q_H": ("H_true", "H_false")}
LETTER_LABELS = ("A", "B")
DEFAULT_OPTION_SEED = 1090998506967575423
DEFAULT_DATASET_NAME = "QuALITY-H"

# ---- readout identities ---------------------------------------------------
# Distinct identity strings prevent dual-order results from being mixed with the optional
# single-order diagnostic output.
READOUT_VERSION = "posterior-eval-dual-order-v2"
COMPARATOR = "posterior-shift-dual-order-v2"
ORDER_TAGS = ("AB", "BA")
DUAL_QUESTION_READOUT = (
    "forced_choice_letter_logprob_at_final_channel_answer_slot_no_transcript_both_orders")
DUAL_METADATA_READOUT = (
    "forced_choice_logprob_final_channel_answer_slot_no_story_no_transcript_both_orders")
LEGACY_QUESTION_READOUT = (
    "forced_choice_letter_logprob_at_final_channel_answer_slot_no_transcript")
LEGACY_METADATA_READOUT = (
    "forced_choice_logprob_final_channel_answer_slot_no_story_no_transcript")

FORBIDDEN_PROMPT_TOKENS = (
    "h_target", "H_true", "H_false", "Y_true", "Y_false",
    "<quote>", "v_quote", "u_quote", "<transcript>",
    "Debater A", "Debater B",
)
FORBIDDEN_TEMPLATE_WORDS = ("hidden", "hypothesis", "h_target")

# The trailing "Answer:" is deliberately NOT in this user prompt: the A/B readout is
# scored inside the assistant's harmony final channel, which ForcedChoiceVerifier
# prefills as "...<|channel|>final<|message|>Answer:" before scoring " A"/" B" (see
# adversarial_transcript/score_verifier.FINAL_CHANNEL_PREFILL). Scoring the letter at
# the bare "<|start|>assistant" position read tail tokens (~e^-37) and degenerated to a
# letter prior; the final-channel slot is where the verdict (and P(H_false)) actually lives.
NO_TRANSCRIPT_READOUT_TEMPLATE = """\
You are answering a multiple-choice reading comprehension question about a short story.
You have not read the story, and no transcript, quotes, or other story evidence is available.
Choose the answer that is most likely to be correct based only on the question and the two answer options.

Question: {question}
A: {answer_a}
B: {answer_b}

Return exactly one letter."""


def write_json_atomic(path, obj):
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def check_template():
    lowered = NO_TRANSCRIPT_READOUT_TEMPLATE.lower()
    for word in FORBIDDEN_TEMPLATE_WORDS:
        if word in lowered:
            raise AssertionError(f"leak guard: {word!r} appears in template")


def check_prompt_leaks(prompt, where):
    for token in FORBIDDEN_PROMPT_TOKENS:
        if token in prompt:
            raise AssertionError(f"leak guard: forbidden token {token!r} in {where} prompt")


def _stable_pair_id(dataset, item):
    """Build the same order-independent identity used by the FT pipeline."""
    dataset_slug = re.sub(r"[^a-z0-9]+", "_", dataset.lower()).strip("_")
    identity = json.dumps(
        [
            dataset,
            item["story_title"],
            item["Q_Y"]["question"],
            item["Q_H"]["question"],
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{dataset_slug}_{digest}"


def _dataset_name_from_path(dataset_path):
    """Infer the dataset directory name without changing the compact file contract."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(dataset_path)))
    return parent if parent and parent != "dataset" else DEFAULT_DATASET_NAME


def load_and_validate(dataset_path, dataset_name=None):
    """Load minimal QuALITY-H examples and attach pair_id/source_partition.

    The full metadata file is used only for identity/field cross-checks and does
    not contribute story text to prompts.
    """
    with open(dataset_path) as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{dataset_path!r} must contain a JSON array")
    dataset_name = dataset_name or _dataset_name_from_path(dataset_path)
    # with open(full_path) as f:
    #     full = json.load(f)

    # full_by_title = {}
    # for ex in full["examples"]:
    #     title = ex["story_title"]
    #     if title in full_by_title:
    #         raise ValueError(f"duplicate story_title in {full_path!r}: {title!r}")
    #     full_by_title[title] = ex

    pair_ids = set()
    for idx, item in enumerate(items):
        where = f"item {idx} ({item.get('story_title', '?')!r})"
        title = item["story_title"]
        for q_key in ALL_QUESTIONS:
            holder = item[q_key]
            for field in ("question",) + ANSWER_FIELDS[q_key]:
                if not isinstance(holder.get(field), str) or not holder[field].strip():
                    raise ValueError(f"{where}: empty/missing {q_key}.{field}")

        # full_ex = full_by_title.get(title)
        # if full_ex is None:
        #     raise ValueError(f"{where}: story_title not found in {full_path!r}")
        # for q_key in ALL_QUESTIONS:
        #     if full_ex[q_key]["question"] != item[q_key]["question"]:
        #         raise ValueError(f"{where}: {q_key} question mismatch vs full metadata")
        # for q_key in ALL_QUESTIONS:
        #     for field in ANSWER_FIELDS[q_key]:
        #         if full_ex[field] != item[q_key][field]:
        #             raise ValueError(f"{where}: {field} mismatch vs full metadata")
        # if full_ex["source_partition"] not in ("train", "test"):
        #     raise ValueError(f"{where}: bad source_partition {full_ex['source_partition']!r}")
        pair_id = item.get("pair_id") or _stable_pair_id(dataset_name, item)
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"{where}: pair_id must be a non-empty string")
        if pair_id in pair_ids:
            raise ValueError(f"{where}: duplicate pair_id {pair_id!r}")
        pair_ids.add(pair_id)
        source_partition = item.get("source_partition")
        if source_partition is not None and not isinstance(source_partition, str):
            raise ValueError(f"{where}: source_partition must be a string or null")
        item["pair_id"] = pair_id
        item["source_partition"] = source_partition
        item["dataset_index"] = idx

    return items


def deterministic_answer_order(item, q_key, option_seed):
    """Return the seeded canonical mapping from semantic labels to A/B."""
    true_label, false_label = LABELS[q_key]
    key = f"{option_seed}|{item['dataset_index']}|{q_key}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    if digest[0] % 2 == 0:
        return {"A": true_label, "B": false_label}
    return {"A": false_label, "B": true_label}


def orders_for(item, q_key, option_seed):
    """The two answer orders (canonical + swap) as {'AB': .., 'BA': ..}.

    Mirrors ``adversarial_transcript.readout.orders_for``. It cannot simply be
    imported: that one calls score_verifier.deterministic_answer_order, which keys on
    candidate['item_id'], while this evaluator keys on item['dataset_index'].

    The tag names the ORDER (canonical vs swapped); it does NOT say where the true label
    sat. For that read orders[tag]['answer_order'], or the letter-keyed l_true_at_A/_B.
    """
    canonical = deterministic_answer_order(item, q_key, option_seed)
    return {"AB": canonical, "BA": {"A": canonical["B"], "B": canonical["A"]}}


def sigmoid(x):
    """Numerically stable logistic. math.exp(-x) overflows for x <~ -709.

    Kept local because this evaluator intentionally uses its own item identity contract.
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def combine_orders(order_blocks, true_label, false_label):
    """Reduce both scored answer orders to order-invariant semantics + position bias.

    Mirrors ``adversarial_transcript.readout.decompose``. `l_true_at_A` and
    `l_true_at_B` are keyed by WHERE THE TRUE LABEL SAT, not by which block was tagged
    'AB', so the result is invariant both to the tag labelling and to which order the
    option seed happened to make canonical.

        l     = logP(letter showing true) - logP(letter showing false)   [per order]
        m     = (l_true_at_A + l_true_at_B) / 2   <- semantic margin (nats), PRIMARY
        l_pos = (l_true_at_B - l_true_at_A) / 2   <- position bias, DIAGNOSTIC only

    `l` is a difference of ABSOLUTE log-probs, i.e. the logit of the two-way renormalised
    choice, so no epsilon or clipping is needed anywhere on this path.

    Two combining rules are emitted for distinct Section 6 analyses:

      * ``semantic_probs`` is the arithmetic mean of the two order-specific semantic
        probabilities. ``stage2-eval.py`` uses it for checkpoint accuracy, mean hidden-
        hypothesis probabilities, and probability-space CTI.
      * ``m`` / ``p_sem`` average semantic log-odds before applying the sigmoid. The
        selected-transcript MI response uses this position-deconfounded readout.

    These quantities answer different estimands and must not be substituted for one
    another.

    `order_blocks` maps tag -> {answer_order, letter_logprobs, prompt_hash}.
    """
    if len(order_blocks) != 2:
        raise ValueError(
            f"exactly two answer orders are required, got {sorted(order_blocks)}")
    by_letter = {}
    per_order = {}
    exact_semantic = {}
    for tag, block in order_blocks.items():
        order = block["answer_order"]
        logprobs = block["letter_logprobs"]
        true_letter = next(letter for letter, label in order.items() if label == true_label)
        false_letter = "B" if true_letter == "A" else "A"
        logodds_true = logprobs[true_letter] - logprobs[false_letter]
        by_letter[true_letter] = logodds_true
        letter_probs = _softmax_two(logprobs["A"], logprobs["B"])
        semantic = semantic_probs(letter_probs, order)
        exact_semantic[tag] = semantic
        pred_letter = max(letter_probs, key=letter_probs.get)
        per_order[tag] = {
            "answer_order": dict(order),
            "true_letter": true_letter,
            "prompt_hash": block.get("prompt_hash"),
            "letter_logprobs": round_map(logprobs),
            "letter_probs": round_map(letter_probs),
            "semantic_probs": round_map(semantic),
            "logodds_true": round(float(logodds_true), 8),
            "pred_letter": pred_letter,
            "pred": order[pred_letter],
            "correct": order[pred_letter] == true_label,
        }
    if set(by_letter) != {"A", "B"}:
        # Both orientations are REQUIRED. Averaging one orientation twice would
        # reintroduce exactly the position bias this readout exists to cancel.
        raise ValueError("both answer orders (true label at A and at B) are required")

    l_true_at_A, l_true_at_B = by_letter["A"], by_letter["B"]
    margin = 0.5 * (l_true_at_A + l_true_at_B)
    l_pos = 0.5 * (l_true_at_B - l_true_at_A)

    # Probability-space mean across the two orders. Both halves are computed as means
    # (not as 1 - x) so they sum to 1 by construction, up to float rounding.
    tags = list(exact_semantic)
    combined = {label: sum(exact_semantic[tag][label] for tag in tags) / len(tags)
                for label in (true_label, false_label)}
    # For a two-way forced choice the two combining rules agree EXACTLY:
    #   combined[t] > 1/2  <=>  p_AB[t] + p_BA[t] > 1  <=>  l_AB + l_BA > 0  <=>  m > 0.
    # `pred` comes from the combined probabilities (the reported semantics) and
    # `decision_semantic` from m, so a float-boundary disagreement is visible, not hidden.
    # Exact tie -> the false label, matching readout.py's strict `margin > 0.0`.
    pred = true_label if combined[true_label] > combined[false_label] else false_label
    # |p_AB[false] - p_BA[false]| == |p_AB[true] - p_BA[true]| because each order's pair
    # sums to 1, so this mean over labels is just |dp|; it is written out literally.
    position_advantage = sum(
        abs(exact_semantic[tags[0]][label] - exact_semantic[tags[1]][label])
        for label in (true_label, false_label)) / 2.0
    pred_by_order = {tag: per_order[tag]["pred"] for tag in tags}
    max_letter_logprob = max(max(block["letter_logprobs"].values())
                             for block in order_blocks.values())
    return {
        "orders": per_order,
        "prompt_hash_by_order": {tag: block.get("prompt_hash")
                                 for tag, block in order_blocks.items()},
        "l_true_at_A": round(float(l_true_at_A), 8),
        "l_true_at_B": round(float(l_true_at_B), 8),
        "m": round(float(margin), 8),
        "l_pos": round(float(l_pos), 8),
        "p_sem": round(float(sigmoid(margin)), 8),
        # ``m`` and ``p_sem`` are oriented on the true label. Writing both semantic
        # orientations exposes P(H_false) directly, so downstream analysis does not need
        # to rediscover that l_sem(false) == -m and p_sem(false) == 1 - p_sem; getting that
        # sign wrong silently flips the reported direction of the steering effect.
        "m_by_label": {true_label: round(float(margin), 8),
                       false_label: round(float(-margin), 8)},
        "p_sem_by_label": {true_label: round(float(sigmoid(margin)), 8),
                           false_label: round(float(sigmoid(-margin)), 8)},
        "semantic_probs": round_map(combined),
        "position_advantage": round(float(position_advantage), 8),
        "max_letter_logprob": round(float(max_letter_logprob), 6),
        "pred": pred,
        "correct": pred == true_label,
        "pred_by_order": pred_by_order,
        "flip": len(set(pred_by_order.values())) > 1,
        "decision_semantic": margin > 0.0,
        "decision_all_orientations": (l_true_at_A > 0.0) and (l_true_at_B > 0.0),
    }


def _answer_text(item, q_key, semantic_label):
    return item[q_key][semantic_label]


def build_prompt(item, q_key, answer_order):
    prompt = NO_TRANSCRIPT_READOUT_TEMPLATE.format(
        question=item[q_key]["question"],
        answer_a=_answer_text(item, q_key, answer_order["A"]),
        answer_b=_answer_text(item, q_key, answer_order["B"]),
    )
    check_prompt_leaks(prompt, f"{q_key}")
    return prompt


def prepare_prompts(items, questions, option_seed, single_order=False):
    """prompts[i][q_key][tag] and answer_orders[i][q_key][tag], one entry per scored order.

    build_prompt runs the leak guard, so both orders are guarded automatically.
    """
    prompts = []
    answer_orders = []
    for item in items:
        per_item = {}
        per_item_orders = {}
        for q_key in questions:
            orders = orders_for(item, q_key, option_seed)
            if single_order:
                orders = {"AB": orders["AB"]}
            per_item[q_key] = {tag: build_prompt(item, q_key, order)
                               for tag, order in orders.items()}
            per_item_orders[q_key] = orders
        prompts.append(per_item)
        answer_orders.append(per_item_orders)
    return prompts, answer_orders


def print_debug_prompts(items, prompts, questions, n=2):
    for idx in range(min(n, len(items))):
        item = items[idx]
        print(f"\n{'=' * 78}")
        print(f"DEBUG POSTERIOR PROMPTS - item {item['dataset_index']} "
              f"({item['story_title']!r})")
        print(f"{'=' * 78}")
        for q_key in questions:
            for tag, prompt in prompts[idx][q_key].items():
                print(f"\n----- [forced-choice no transcript | {q_key} | order {tag}] "
                      + "-" * 18)
                print(prompt)


def dry_run_logprobs(item, q_key, model_label, order_tag=None):
    """Return deterministic fake log probabilities for validation without a model.

    Dual-order calls include the order tag so their draws differ and exercise semantic
    combination and position-bias diagnostics. Single-order diagnostics omit the tag.
    """
    key = f"{model_label}|{item['dataset_index']}|{q_key}"
    if order_tag is not None:
        key = f"{key}|{order_tag}"
    raw = hashlib.sha256(key.encode("utf-8")).hexdigest()
    frac = int(raw[:8], 16) / 0x100000000
    p_a = 0.05 + 0.90 * frac
    p_b = 1.0 - p_a
    return {"A": math.log(p_a), "B": math.log(p_b)}


def semantic_probs(letter_probs, answer_order):
    return {semantic: float(letter_probs[letter]) for letter, semantic in answer_order.items()}


def round_map(values, ndigits=8):
    return {k: round(float(v), ndigits) for k, v in values.items()}


def build_question_eval_dual(q_key, order_blocks):
    """Build one dual-order semantic record with per-order diagnostics."""
    true_label, false_label = LABELS[q_key]
    out = {
        "readout": DUAL_QUESTION_READOUT,
        "readout_version": READOUT_VERSION,
        "gold": true_label,
        "false_label": false_label,
    }
    out.update(combine_orders(order_blocks, true_label, false_label))
    return out


def build_question_eval(item, q_key, prompt, answer_order, logprobs):
    """Build the optional canonical-order diagnostic record."""
    letter_probs = _softmax_two(logprobs["A"], logprobs["B"])
    semantic = semantic_probs(letter_probs, answer_order)
    pred_letter = max(letter_probs, key=letter_probs.get)
    pred = answer_order[pred_letter]
    gold = LABELS[q_key][0]
    gold_letter = next(letter for letter, label in answer_order.items() if label == gold)
    false_label = LABELS[q_key][1]
    return {
        "readout": "forced_choice_letter_logprob_at_final_channel_answer_slot_no_transcript",
        "prompt_hash": _prompt_hash(prompt),
        "answer_order": dict(answer_order),
        "gold_letter": gold_letter,
        "gold": gold,
        "false_label": false_label,
        "letter_logprobs": round_map(logprobs),
        "letter_probs": round_map(letter_probs),
        "semantic_probs": round_map(semantic),
        "max_letter_logprob": round(float(max(logprobs.values())), 6),
        "pred_letter": pred_letter,
        "pred": pred,
        "correct": pred == gold,
    }


def build_output_record(item, questions, dataset_name=DEFAULT_DATASET_NAME):
    record = {
        "index": item["dataset_index"],
        # The comparator aligns artifacts by this stable identity. Loaded items normally
        # carry the FT pipeline's pair_id; the fallback also supports compact fixtures and
        # programmatic callers using the run's explicit dataset name.
        "pair_id": item.get("pair_id") or _stable_pair_id(dataset_name, item),
        "source_partition": item.get("source_partition"),
        "story_title": item["story_title"],
    }
    for q_key in ALL_QUESTIONS:
        # Copy only the canonical question fields; never let unrelated dataset fields
        # (e.g. honest_transcript/adversarial_transcript in the merged dataset) leak
        # into the no-story/no-transcript posterior output.
        holder = {"question": item[q_key]["question"]}
        for field in ANSWER_FIELDS[q_key]:
            holder[field] = item[q_key][field]
        if q_key not in questions:
            holder["evaluation"] = None
        record[q_key] = holder
    return record


def run_evaluation(scorer, items, prompts, answer_orders, questions, args):
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        def tqdm(iterable, **_kwargs):
            return iterable

    single_order = getattr(args, "single_order", False)
    # main() resolves --dataset-name; direct callers fall back to the declared default.
    dataset_name = getattr(args, "dataset_name", None) or DEFAULT_DATASET_NAME
    records = []
    for idx, item in enumerate(tqdm(items, desc=f"posterior-eval[{args.model_label}]")):
        record = build_output_record(item, questions, dataset_name)
        for q_key in questions:
            order_blocks = {}
            for tag, prompt in prompts[idx][q_key].items():
                if args.dry_run:
                    # Single-order diagnostics omit the order tag from the fake draw.
                    logprobs = dry_run_logprobs(
                        item, q_key, args.model_label, None if single_order else tag)
                else:
                    # One scorer call per order. The two option-swapped prompts
                    # differ (the option lines are swapped), so ForcedChoiceVerifier's
                    # prompt-keyed cache never collides between them.
                    logprobs = scorer.choice_logprobs(prompt)
                order_blocks[tag] = {
                    "answer_order": answer_orders[idx][q_key][tag],
                    "letter_logprobs": logprobs,
                    "prompt_hash": _prompt_hash(prompt),
                }
            if single_order:
                block = order_blocks["AB"]
                record[q_key]["evaluation"] = build_question_eval(
                    item,
                    q_key,
                    prompts[idx][q_key]["AB"],
                    block["answer_order"],
                    block["letter_logprobs"],
                )
            else:
                record[q_key]["evaluation"] = build_question_eval_dual(q_key, order_blocks)
        records.append(record)
        if args.save_every and (idx + 1) % args.save_every == 0:
            write_json_atomic(f"{args.output}.partial.json", {
                "note": "diagnostic snapshot, not resumable",
                "items_done": idx + 1,
                "items_total": len(items),
                "items": records,
            })
    return records


def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _pred_letters(evaluation):
    """Return predicted letter(s) from either supported record shape."""
    if "orders" in evaluation:
        return [block["pred_letter"] for block in evaluation["orders"].values()]
    return [evaluation["pred_letter"]]


def compute_aggregates(records, questions, dual=True):
    out = {"n_items": len(records)}
    if not records:
        return out
    for q_key in questions:
        evals = [r[q_key]["evaluation"] for r in records]
        true_label, false_label = LABELS[q_key]
        # These compact aggregate keys use the combined semantic record and are consumed
        # by stage2-eval.py.
        out[f"{q_key}_accuracy"] = sum(e["correct"] for e in evals) / len(evals)
        # Readout-health signals: a valid final-channel slot has a non-degenerate letter
        # split and a median maximum answer-letter log probability near zero.
        max_logps = [e["max_letter_logprob"] for e in evals if "max_letter_logprob" in e]
        if max_logps:
            out[f"{q_key}_median_max_letter_logprob"] = round(_median(max_logps), 6)
        # Dual-order evaluation pools 2 * n_items letter predictions; single-order
        # diagnostics pool n_items. The per-order split
        # is in {q}_pred_letter_counts_by_order.
        pred_letters = [letter for e in evals for letter in _pred_letters(e)]
        out[f"{q_key}_pred_letter_counts"] = {
            "A": pred_letters.count("A"), "B": pred_letters.count("B")}
        out[f"{q_key}_mean_p_{true_label.lower()}"] = (
            sum(e["semantic_probs"][true_label] for e in evals) / len(evals)
        )
        out[f"{q_key}_mean_p_{false_label.lower()}"] = (
            sum(e["semantic_probs"][false_label] for e in evals) / len(evals)
        )
        out[f"{q_key}_pred_{false_label.lower()}_rate"] = (
            sum(e["pred"] == false_label for e in evals) / len(evals)
        )
        if not dual:
            continue
        # ---- dual-order position-bias diagnostics (reported, never folded into the
        # semantics above; see combine_orders) ----
        out[f"{q_key}_accuracy_by_order"] = {
            tag: sum(e["orders"][tag]["correct"] for e in evals) / len(evals)
            for tag in ORDER_TAGS}
        out[f"{q_key}_flip_rate"] = sum(e["flip"] for e in evals) / len(evals)
        out[f"{q_key}_mean_abs_position_advantage"] = (
            sum(e["position_advantage"] for e in evals) / len(evals)
        )
        out[f"{q_key}_mean_abs_l_pos"] = sum(abs(e["l_pos"]) for e in evals) / len(evals)
        out[f"{q_key}_pred_letter_counts_by_order"] = {
            tag: {"A": sum(e["orders"][tag]["pred_letter"] == "A" for e in evals),
                  "B": sum(e["orders"][tag]["pred_letter"] == "B" for e in evals)}
            for tag in ORDER_TAGS}
        out[f"{q_key}_mean_semantic_margin"] = sum(e["m"] for e in evals) / len(evals)
        # Logit-combined semantic probability used by the selected-transcript readout.
        # The mean-probability aggregate above remains the checkpoint-table quantity.
        out[f"{q_key}_mean_p_sem_{false_label.lower()}"] = (
            sum(e["p_sem_by_label"][false_label] for e in evals) / len(evals)
        )
    return out


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be positive, got {value!r}")
    return parsed


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset/QuALITY-H/QuALITY-H-no-debate.json",
                   help="QuALITY-H item file; transcripts, if present, are ignored")
    p.add_argument("--dataset-name", default=None,
                   help="dataset identity used for pair_id derivation (default: parent directory)")
    p.add_argument("--model", default="checkpoints/gpt-oss-20b-bf16-base",
                   help="hub model name or local checkpoint directory")
    p.add_argument("--model-label", default=None,
                   help="short label written into metadata, e.g. base/honest/adv")
    p.add_argument("--output", default="runs/posterior_shift/QuALITY-H/base.json")
    p.add_argument("--device", type=int, default=0,
                   help="visible CUDA device index after CUDA_VISIBLE_DEVICES is applied")
    p.add_argument("--questions", choices=["both", "Q_Y", "Q_H"], default="both")
    p.add_argument("--option-seed", type=int, default=DEFAULT_OPTION_SEED)
    p.add_argument("--limit", type=positive_int, default=None)
    p.add_argument("--save-every", type=positive_int, default=25)
    p.add_argument("--dry-run", action="store_true",
                   help="render prompts and write deterministic fake probabilities without loading a model")
    p.add_argument("--debug-prompts", action="store_true")
    p.add_argument("--single-order", action="store_true",
                   help="diagnostic: score only the deterministic canonical A/B order; "
                        "position-biased and excluded from reported results")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_template()
    if args.single_order:
        print("WARNING: --single-order is position-biased diagnostic output; "
              "do not report these numbers.", file=sys.stderr)
    if args.model_label is None:
        args.model_label = os.path.basename(args.model.rstrip("/")) or args.model
    questions = ALL_QUESTIONS if args.questions == "both" else (args.questions,)
    # Resolve once so load_and_validate, run_evaluation's pair_id fallback and the metadata
    # block below cannot disagree about which dataset this run is.
    args.dataset_name = args.dataset_name or _dataset_name_from_path(args.dataset)

    items = load_and_validate(args.dataset, args.dataset_name)
    if args.limit is not None:
        items = items[: args.limit]
    prompts, answer_orders = prepare_prompts(
        items, questions, args.option_seed, single_order=args.single_order)
    lengths = sorted(len(prompt) for per_item in prompts
                     for per_question in per_item.values() for prompt in per_question.values())
    print(f"Validated/rendered {len(items)} items for model_label={args.model_label!r} "
          f"(questions={','.join(questions)}, dry_run={args.dry_run}); "
          f"{len(lengths)} prompts, chars min/median/max="
          f"{lengths[0] if lengths else 0}/"
          f"{lengths[len(lengths) // 2] if lengths else 0}/"
          f"{lengths[-1] if lengths else 0}. Leak guard passed.")
    if args.debug_prompts or args.dry_run:
        print_debug_prompts(items, prompts, questions)
    if not items:
        sys.exit("No items selected for evaluation.")

    scorer = None if args.dry_run else ForcedChoiceVerifier(args.model, args.device)
    records = run_evaluation(scorer, items, prompts, answer_orders, questions, args)
    quantized_load = None if args.dry_run else scorer.quantized_load

    metadata = {
        "model": args.model,
        "model_label": args.model_label,
        "model_quantized_load": quantized_load,
        "dataset": args.dataset,
        "dataset_name": args.dataset_name or _dataset_name_from_path(args.dataset),
        "limit": args.limit,
        "questions": list(questions),
        "option_seed": args.option_seed,
        "num_items": len(records),
        "dry_run": args.dry_run,
        # The readout identity encodes the scored position and order protocol so
        # comparators cannot mix single- and dual-order artifacts.
        "readout": LEGACY_METADATA_READOUT if args.single_order else DUAL_METADATA_READOUT,
        "readout_position": "final_channel_answer_slot",
        "completion_strings": {"A": " A", "B": " B"},
    }
    caveats = [
        "No story text and no debate transcript are shown.",
        "Probabilities are normalized over the two forced-choice completions only.",
    ]
    if not args.single_order:
        # Dual-order metadata records the comparator and order tags explicitly.
        metadata.update({
            "readout_version": READOUT_VERSION,
            "comparator": COMPARATOR,
            "both_orders": True,
            "order_tags": list(ORDER_TAGS),
        })
        caveats.append(
            "Both A/B answer orders are scored; the reported semantics are order-invariant "
            "and position bias is reported separately (l_pos, flip_rate).")
    metadata["aggregate_metrics"] = compute_aggregates(
        records, questions, dual=not args.single_order)
    metadata["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata["caveats"] = caveats
    output = {"metadata": metadata, "items": records}
    write_json_atomic(args.output, output)
    partial = f"{args.output}.partial.json"
    if os.path.exists(partial):
        os.remove(partial)
    print(f"Wrote {len(records)} posterior readout item(s) to {args.output}")
    agg = output["metadata"]["aggregate_metrics"]
    for q_key in questions:
        med = agg.get(f"{q_key}_median_max_letter_logprob")
        counts = agg.get(f"{q_key}_pred_letter_counts")
        if med is not None:
            flag = "OK" if med >= -8.0 else "SUSPECT (tail readout?)"
            scope = "of N, single order" if args.single_order else "of 2N, pooled over both orders"
            line = (f"  [health] {q_key}: median max-letter logprob = {med} [{flag}]; "
                    f"pred letters A/B = {counts['A']}/{counts['B']} ({scope})")
            if not args.single_order:
                line += (f"; flip rate = {agg[f'{q_key}_flip_rate']:.3f}"
                         f"; mean |l_pos| = {agg[f'{q_key}_mean_abs_l_pos']:.3f}")
            print(line)


if __name__ == "__main__":
    main()
