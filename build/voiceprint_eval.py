"""Voiceprint enrollment and open-set matching on far-field meeting audio (AMI corpus).

    python voiceprint_eval.py embed     # one ECAPA embedding per utterance, cached to disk
    python voiceprint_eval.py evaluate  # enroll, match, write results/metrics.json

Setup: the AMI test set has four series of four meetings (EN2002, ES2004, IS1009, TS3003), each
with the same four people. Meetings a and b are past conversations used for enrollment; c and d
are later conversations used for testing. In each of four folds, one person from every series is
never enrolled, so the wearer's roster holds 12 people and 4 people speak as strangers.

Test audio is the single distant table microphone (far-field). A noisy copy of the test audio
mixes in three other talkers from other meetings at 5 dB and 0 dB, like a louder table nearby.
"""
import argparse
import glob
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SAMPLE_RATE = 16000
MIN_SEC = 1.0  # shorter utterances carry too little voice to embed
MAX_SEC = 15.0
ENROLL_MEETINGS = ("a", "b")
TEST_MEETINGS = ("c", "d")
MIN_PRECISION = 0.99  # at most 1 wrong name per 100 names shown
MIN_REJECTED = 0.99  # at most 1 stranger in 100 gets a name
MARGIN = 0.05  # best score must beat the runner-up by this much
BABBLE_SNR_DB = (5, 0)
BABBLE_TALKERS = 3
BABBLE_POOL = 300
SEED = 0


def read_rows(mic_dir, columns):
    import pyarrow.parquet as pq

    for path in sorted(glob.glob(f"{mic_dir}/*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


def overlapped_keys(rows):
    """Utterances that overlap in time with another speaker's utterance in the same meeting."""
    by_meeting = defaultdict(list)
    for r in rows:
        by_meeting[r["meeting_id"]].append((r["begin_time"], r["end_time"], r["speaker_id"]))
    keys = set()
    for meeting, items in by_meeting.items():
        items.sort()
        for i, (b, e, s) in enumerate(items):
            for b2, _, s2 in items[i + 1:]:
                if b2 >= e:
                    break
                if s2 != s:
                    keys.add((meeting, s, b))
                    keys.add((meeting, s2, b2))
    return keys


def mix_babble(x, talkers, snr_db):
    """Add the sum of other talkers to x at the given speech-to-babble ratio."""
    babble = np.zeros_like(x)
    for t in talkers:
        t = np.resize(t, len(x))  # repeat or cut to the target length
        babble += t / (np.sqrt(np.mean(t ** 2)) + 1e-9)
    speech_power = np.mean(x ** 2)
    babble_power = np.mean(babble ** 2) + 1e-12
    return x + babble * np.sqrt(speech_power / (babble_power * 10 ** (snr_db / 10)))


def embed(args):
    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(DATA / "models" / "ecapa"),
        run_opts={"device": "cpu"},
    )
    assert model is not None, "could not load the ECAPA model"

    def vector(x):
        with torch.no_grad():
            return model.encode_batch(torch.from_numpy(x.astype(np.float32)).unsqueeze(0)).squeeze().numpy()

    def decode(row):
        x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        assert sr == SAMPLE_RATE, sr
        return x[: int(MAX_SEC * sr)]

    def save(name, meta, embs):
        meeting, speaker, begin, dur, ovl = zip(*meta)
        np.savez(args.cache / f"{name}.npz", emb=np.stack(embs), meeting=np.array(meeting),
                 speaker=np.array(speaker), begin=np.array(begin), dur=np.array(dur), overlapped=np.array(ovl))

    args.cache.mkdir(parents=True, exist_ok=True)
    columns = ["meeting_id", "speaker_id", "begin_time", "end_time"]
    if not (args.cache / "sdm.npz").exists():
        overlapped = overlapped_keys(read_rows(args.data / "sdm", columns))
        meta, embs, audio_sec = [], [], 0.0
        for r in read_rows(args.data / "sdm", columns + ["audio"]):
            if r["end_time"] - r["begin_time"] < MIN_SEC:
                continue
            x = decode(r)
            embs.append(vector(x))
            key = (r["meeting_id"], r["speaker_id"], r["begin_time"])
            meta.append((*key, len(x) / SAMPLE_RATE, key in overlapped))
            audio_sec += len(x) / SAMPLE_RATE
        save("sdm", meta, embs)
        print(f"sdm: {len(meta)} utterances, {audio_sec / 3600:.1f} h")

    names = [f"sdm_babble{snr}" for snr in BABBLE_SNR_DB]
    if not all((args.cache / f"{n}.npz").exists() for n in names):
        rows = list(read_rows(args.data / "sdm", columns))
        overlapped = overlapped_keys(rows)
        rng = np.random.default_rng(SEED)
        long_enough = [i for i, r in enumerate(rows) if r["end_time"] - r["begin_time"] >= 3.0]
        pool_ids = set(rng.choice(long_enough, size=BABBLE_POOL, replace=False).tolist())
        pool = []  # (series, audio)
        for i, r in enumerate(read_rows(args.data / "sdm", columns + ["audio"])):
            if i in pool_ids:
                pool.append((r["meeting_id"][:-1], decode(r)))
        out = {snr: ([], []) for snr in BABBLE_SNR_DB}
        for i, r in enumerate(read_rows(args.data / "sdm", columns + ["audio"])):
            if r["end_time"] - r["begin_time"] < MIN_SEC or r["meeting_id"][-1] not in TEST_MEETINGS:
                continue
            x = decode(r)
            others = [a for s, a in pool if s != r["meeting_id"][:-1]]
            pick = np.random.default_rng(SEED + i).choice(len(others), size=BABBLE_TALKERS, replace=False)
            key = (r["meeting_id"], r["speaker_id"], r["begin_time"])
            for snr in BABBLE_SNR_DB:
                out[snr][0].append((*key, len(x) / SAMPLE_RATE, key in overlapped))
                out[snr][1].append(vector(mix_babble(x, [others[j] for j in pick], snr)))
        for snr, (meta, embs) in out.items():
            save(f"sdm_babble{snr}", meta, embs)
        print("babble copies written")


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def pooled(emb, dur, idx):
    """Duration-weighted mean of utterance embeddings: the voiceprint of that stretch of speech."""
    return unit((emb[idx] * dur[idx, None]).sum(axis=0))


def windows(idx, dur, seconds):
    """Split time-ordered utterances into consecutive groups holding about `seconds` of speech.
    seconds=None keeps one utterance per group."""
    if seconds is None:
        return [[i] for i in idx]
    groups, cur, total = [], [], 0.0
    for i in idx:
        cur.append(i)
        total += dur[i]
        if total >= seconds:
            groups.append(cur)
            cur, total = [], 0.0
    if total >= 0.8 * seconds:
        groups.append(cur)
    return groups


def decide(scores, threshold, margin):
    """Open-set decision. scores: (trials, roster). Returns the named roster index, or -1 for
    unknown. A name is given only when the best score clears the threshold and beats the
    runner-up by the margin."""
    order = np.argsort(-scores, axis=1)
    rows = np.arange(len(scores))
    best = scores[rows, order[:, 0]]
    second = scores[rows, order[:, 1]] if scores.shape[1] > 1 else np.full(len(scores), -np.inf)
    return np.where((best >= threshold) & (best - second >= margin), order[:, 0], -1)


def rates(pred, truth):
    """truth: roster index of the real speaker, or -1 when the speaker is not on the roster."""
    known, named = truth >= 0, pred >= 0
    correct = named & (pred == truth)
    return {
        "name_precision": correct.sum() / max(named.sum(), 1),
        "known_recall": correct[known].sum() / max(known.sum(), 1),
        "unknown_rejected": (~named[~known]).sum() / max((~known).sum(), 1),
    }


def load(cache, name):
    z = np.load(cache / f"{name}.npz")
    d = {k: z[k] for k in z.files}
    d["emb"] = unit(d["emb"])
    d["part"] = np.array([m[-1] for m in d["meeting"]])
    return d


def speaker_index(d, parts, clean_only):
    """(meeting, speaker) -> time-ordered utterance indices."""
    out = defaultdict(list)
    for i in np.argsort(d["begin"], kind="stable"):
        if d["part"][i] in parts and not (clean_only and d["overlapped"][i]):
            out[(d["meeting"][i], d["speaker"][i])].append(i)
    return dict(sorted(out.items()))


def enrollment_prints(enroll, enroll_sec):
    """One voiceprint per person from their clean speech in meetings a then b, capped at enroll_sec."""
    by_person = defaultdict(list)
    for (_, spk), idx in speaker_index(enroll, ENROLL_MEETINGS, clean_only=True).items():
        by_person[spk].extend(idx)
    if enroll_sec is not None:
        by_person = {spk: first_seconds(idx, enroll["dur"], enroll_sec) for spk, idx in by_person.items()}
    return {spk: pooled(enroll["emb"], enroll["dur"], idx) for spk, idx in by_person.items()}


def folds(enroll):
    """Four folds; in fold f the f-th person of every series is the stranger."""
    people = defaultdict(set)
    for m, s in zip(enroll["meeting"], enroll["speaker"]):
        people[m[:-1]].add(s)
    return [{sorted(p)[f] for p in people.values()} for f in range(4)]


def trials(enroll, test, window_sec, enroll_sec: float | None = 60.0, test_clean_only=True):
    """Scores of every test unit against the roster, stacked over folds.
    Returns (scores, truth, units_per_fold, series of each row); truth is the roster index or -1
    for a stranger."""
    prints = enrollment_prints(enroll, enroll_sec)
    units = []  # (series, speaker, voiceprint)
    for (meeting, spk), idx in speaker_index(test, TEST_MEETINGS, test_clean_only).items():
        for group in windows(idx, test["dur"], window_sec):
            units.append((meeting[:-1], spk, pooled(test["emb"], test["dur"], group)))
    unit_vp = np.stack([v for _, _, v in units])
    all_scores, all_truth = [], []
    fold_list = folds(enroll)
    for strangers in fold_list:
        roster = sorted(set(prints) - strangers)
        all_scores.append(unit_vp @ np.stack([prints[p] for p in roster]).T)
        all_truth.append([roster.index(s) if s in roster else -1 for _, s, _ in units])
    series = np.tile([u[0] for u in units], len(fold_list))
    return np.concatenate(all_scores), np.concatenate(all_truth), len(units), series


def operating_point(scores, truth, min_precision=MIN_PRECISION, min_rejected=MIN_REJECTED):
    """Lowest threshold whose name precision and stranger rejection reach their targets
    (the lowest such threshold keeps the most enrolled people named)."""
    for t in np.arange(0.0, 1.0, 0.005):
        r = rates(decide(scores, t, MARGIN), truth)
        if r["name_precision"] >= min_precision and r["unknown_rejected"] >= min_rejected:
            return float(t), r
    return None, None


def held_out(scores, truth, series):
    """Pick the threshold on the people of two series, measure it on the other two, then swap.
    The reported rates never use the windows that chose their threshold."""
    names = sorted(set(series))
    halves = (names[:2], names[2:])
    pred, thresholds = np.full(len(truth), -1), []
    for tune, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        tm, xm = np.isin(series, tune), np.isin(series, test)
        thr, _ = operating_point(scores[tm], truth[tm])
        assert thr is not None, "no threshold meets the targets"
        thresholds.append(thr)
        pred[xm] = decide(scores[xm], thr, MARGIN)
    return {"thresholds": thresholds, **{k: float(v) for k, v in rates(pred, truth).items()}}


def first_seconds(idx, dur, seconds):
    keep, total = [], 0.0
    for i in idx:
        if total >= seconds:
            break
        keep.append(i)
        total += dur[i]
    return keep


def pieces(idx, dur, piece_sec=10.0, max_pieces=6):
    """A label's clean speech as at most 6 pieces of 10 s, spread over the whole label, so long and
    short labels are handled alike. The design pools these pieces and checks them."""
    groups = windows(idx, dur, piece_sec)
    if len(groups) > max_pieces:
        groups = [groups[i] for i in np.linspace(0, len(groups) - 1, max_pieces).round().astype(int)]
    return groups


def consistency(emb, dur, idx, piece_sec=10.0, max_pieces=6):
    """Lowest similarity between any two pieces of one label. A label that holds two people,
    because diarization merged them, has a pair of pieces from different voices."""
    vecs = [pooled(emb, dur, g) for g in pieces(idx, dur, piece_sec, max_pieces)]
    if len(vecs) < 2:
        return 1.0
    p = np.stack(vecs)
    return float((p @ p.T)[np.triu_indices(len(p), 1)].min())


def label_decisions(enroll, test, threshold, check):
    """One decision per label, as the design makes it: a person's clean speech in one later meeting
    is one label, pooled from its spread pieces, and named only if it passes the one-voice check."""
    prints = enrollment_prints(enroll, 60.0)
    units = []  # (speaker, voiceprint, passes the check)
    for idx in speaker_index(test, TEST_MEETINGS, clean_only=True).values():
        if test["dur"][idx].sum() < 30.0:
            continue
        flat = [i for g in pieces(idx, test["dur"]) for i in g]
        units.append((test["speaker"][idx[0]], pooled(test["emb"], test["dur"], flat),
                      consistency(test["emb"], test["dur"], idx) >= check))
    vps = np.stack([v for _, v, _ in units])
    passes = np.array([ok for _, _, ok in units])
    preds, before, truths = [], [], []
    for strangers in folds(enroll):
        roster = sorted(set(prints) - strangers)
        pred = decide(vps @ np.stack([prints[p] for p in roster]).T, threshold, MARGIN)
        before.append(pred.copy())
        pred[~passes] = -1
        preds.append(pred)
        truths.append([roster.index(spk) if spk in roster else -1 for spk, _, _ in units])
    truth = np.concatenate(truths)
    out = {k: float(v) for k, v in rates(np.concatenate(preds), truth).items()}
    out["named_before_check"] = float(rates(np.concatenate(before), truth)["known_recall"])
    out["labels"] = len(units)
    out["stranger_decisions"] = int((truth < 0).sum())
    return out


def merge_experiment(enroll, tests):
    """Build labels that hold two people (30 s + 30 s, and 50 s + 10 s) and take real one-person
    labels (all of a person's clean speech in a meeting), and measure the one-voice check on both."""
    singles, merged = defaultdict(list), defaultdict(list)
    for name, d in tests.items():
        by_meeting = defaultdict(dict)
        for (meeting, spk), idx in speaker_index(d, TEST_MEETINGS, clean_only=True).items():
            by_meeting[meeting][spk] = idx
        for spk_idx in by_meeting.values():
            for idx in spk_idx.values():
                if d["dur"][idx].sum() >= 30.0:
                    singles[name].append(consistency(d["emb"], d["dur"], idx))
            people = sorted(spk_idx)
            for i, a in enumerate(people):
                for b in people[i + 1:]:
                    for kind, (sa, sb) in (("even", (30.0, 30.0)), ("uneven", (50.0, 10.0))):
                        ia = first_seconds(spk_idx[a], d["dur"], sa)
                        ib = first_seconds(spk_idx[b], d["dur"], sb)
                        if d["dur"][ia].sum() < 0.8 * sa or d["dur"][ib].sum() < 0.8 * sb:
                            continue
                        merged[(name, kind)].append(consistency(d["emb"], d["dur"], ia + ib))
    # One check threshold for every noise level: the 5th percentile of real one-person labels in
    # quiet audio. In noise it flags more real labels, which only costs names.
    check = float(np.percentile(singles["quiet"], 5))
    out: dict = {"check_threshold": check}
    for name in tests:
        out[name] = {"single_labels": len(singles[name]),
                     "singles_flagged": float((np.array(singles[name]) < check).mean())}
        for kind in ("even", "uneven"):
            m = np.array(merged[(name, kind)])
            out[name][kind] = {"merged_labels": int(len(m)), "caught": float((m < check).mean())}
    return out


def at(scores, truth, threshold):
    return {k: float(v) for k, v in rates(decide(scores, threshold, MARGIN), truth).items()}


def evaluate(args):
    sdm = load(args.cache, "sdm")
    noisy = {snr: load(args.cache, f"sdm_babble{snr}") for snr in BABBLE_SNR_DB}
    tests = {"quiet": sdm, **{f"babble {snr} dB": d for snr, d in noisy.items()}}
    r: dict = {"min_precision": MIN_PRECISION, "min_rejected": MIN_REJECTED, "margin": MARGIN}

    # 1. Threshold: picked on the people of two series, measured on the other two. The design uses
    #    the strictest value any half picked, in any noise condition.
    r["held_out"] = {}
    for name, d in tests.items():
        st, tt, _, ss = trials(sdm, d, 30.0)
        r["held_out"][name] = held_out(st, tt, ss)
    name_thr = max(t for h in r["held_out"].values() for t in h["thresholds"])
    r["name_threshold"] = name_thr

    # 2. The one-voice check, on real one-person labels and on labels that merge two people.
    r["merges"] = merge_experiment(sdm, tests)

    # 3. One decision per label with every gate on, as the design decides.
    r["per_label"] = {name: label_decisions(sdm, d, name_thr, r["merges"]["check_threshold"])
                      for name, d in tests.items()}

    # 4. How much enrollment speech is enough.
    r["enrollment_speech"] = {f"{int(e)} s": at(*trials(sdm, sdm, 30.0, enroll_sec=e)[:2], name_thr)
                              for e in (10.0, 30.0, 60.0)}

    # 5. Overlapped against clean speech, one utterance per decision.
    s_all, t_all, n_units, _ = trials(sdm, sdm, None, test_clean_only=False)
    ovl_units = [bool(sdm["overlapped"][i]) for idx in speaker_index(sdm, TEST_MEETINGS, False).values() for i in idx]
    assert len(ovl_units) == n_units
    ovl = np.tile(np.array(ovl_units), len(t_all) // n_units)
    r["overlap"] = {}
    for label, mask in (("clean", ~ovl), ("overlapped", ovl)):
        known = mask & (t_all >= 0)
        pred = decide(s_all[known], name_thr, MARGIN)
        r["overlap"][label] = {"known_recall": float((pred == t_all[known]).mean()),
                               "wrong_name": float(((pred >= 0) & (pred != t_all[known])).mean())}

    _, t, n, _ = trials(sdm, sdm, 30.0)
    r["data"] = {"utterances": int(len(sdm["dur"])), "hours": float(sdm["dur"].sum() / 3600),
                 "overlapped_share": float(sdm["overlapped"].mean()), "windows_30s": n,
                 "decisions_30s": int(len(t)), "stranger_decisions_30s": int((t < 0).sum())}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(r, indent=2))
    print(json.dumps({k: r[k] for k in ("name_threshold", "per_label")}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["embed", "evaluate"])
    p.add_argument("--data", type=Path, default=DATA / "ami")
    p.add_argument("--cache", type=Path, default=DATA / "embeddings")
    p.add_argument("--out", type=Path, default=HERE / "results")
    args = p.parse_args()
    embed(args) if args.command == "embed" else evaluate(args)


if __name__ == "__main__":
    main()
