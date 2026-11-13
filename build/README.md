# Build: voiceprint matching on far-field audio

**Question.** Can an off-the-shelf speaker embedding model name the people around a wearer across conversations, from far-field audio, without naming strangers? What threshold, and how much speech per decision?

**Approach.**
- **Model:** [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), open source, CPU only, no training.
- **Data:** the [AMI meeting corpus](https://huggingface.co/datasets/edinburghcstr/ami) test set: 16 meetings in four series, with the same four people in every meeting of a series. Audio comes from the single distant microphone on the table.
- **Enrollment and test:** meetings a and b enroll each person from up to 60 s of clean speech. Meetings c and d are the later conversations to recognize them in.
- **Strangers:** in each of four folds, one person from every series is never enrolled. That gives a roster of 12 people and 4 strangers, and the strangers sit in the same rooms as enrolled people.
- **Noise:** a noisy copy of the test audio mixes in three talkers from other meetings at 5 dB and 0 dB, like a louder table nearby.
- **Decision rule:** cosine similarity to each enrolled voiceprint. A name is given only above a threshold and with a margin of 0.05 over the runner-up.

Five experiments, each behind one number in the design ([section 4.6](../docs/system-design.md#46-what-the-build-measured)):

| Experiment | Sets |
|---|---|
| Threshold picked on half the people, tested on the other half | the name threshold, 0.46 |
| One-voice check on real labels and on labels that merge two people | the check threshold, 0.38 |
| One decision per label with every gate on | the results table |
| Enrollment length | 30 s of speech to start a voice |
| Overlapped against clean speech | cutting out overlap |

**Result.**

One decision per label at 0.46 with every gate on (30 labels; 90 decisions about enrolled people and 30 about strangers):

| Condition | Names shown that are right | Enrolled people named | Named before the one-voice check | Strangers left unnamed |
|---|---|---|---|---|
| Quiet room | 100% | 93% | 100% | 100% |
| Talkers at 5 dB | 100% | 50% | 100% | 100% |
| Talkers at 0 dB | 100% | 10% | 77% | 100% |

- **Threshold.** The halves picked 0.435 and 0.38 in a quiet room, 0.44 and 0.34 at 5 dB, 0.46 and 0.35 at 0 dB. A threshold picked on a few people can be too low for the next group, so the design takes the strictest: 0.46.
- **One-voice check.** 0.38 is the 5th percentile of real one-person labels in a quiet room. It rejects 2 of 30 real labels there, 50% at 5 dB and 90% at 0 dB, and catches every even merge and 77% of uneven ones (50 s + 10 s) in a quiet room. Noise costs names, not correctness.
- **Overlapped speech** got the wrong name 2.9% of the time, against 0.1% for clean speech.
- **Enrollment.** With 10 s, enrolled people were named in only 89% of later decisions. 30 s and 60 s named everyone, with no stranger named.

**Lesson.** Pick the threshold on held-out people and take the strictest value. Pool 30 s or more of clean, non-overlapped speech. Refuse to name a label that seems to hold two voices. Hard audio then falls back to "Unknown speaker" instead of a wrong name, at a real cost in names.

Raw numbers: [results/metrics.json](results/metrics.json).

**Limits.**
- AMI's own speaker turns and overlap marks are used, not a diarization system's output, and the merges are simple concatenations.
- The sample is 16 people. With 16 people and no stranger named, a real per-person error rate of up to about 17% cannot be ruled out (95% confidence).
- Enrollment and test share a room, a microphone and a week. The roster has 12 voices, not hundreds.
- The babble is other people talking, not music or wind.

## Run

Embedding takes a few minutes on CPU and downloads about 1.2 GB of audio; evaluation takes under a minute.

```bash
cd build
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m unittest test_voiceprint_eval
mkdir -p ../data/ami/sdm
for i in 0 1 2 3; do
  curl -L -o ../data/ami/sdm/test-0000$i-of-00004.parquet \
    https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/sdm/test-0000$i-of-00004.parquet
done
python voiceprint_eval.py embed
python voiceprint_eval.py evaluate
```
