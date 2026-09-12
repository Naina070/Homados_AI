# ECAPA-TDNN Paper Explained + VoiceGuardAI (SIH 2026, PS 26104) Full Technical Plan

---

# PART 1 — Understanding the ECAPA-TDNN Paper From First Principles

## 1.1 What problem is the paper solving?

The paper is about **speaker verification**: given a short recording of someone's voice, decide whether it belongs to a specific claimed person ("is this really Ramesh speaking, or someone else?"). This is different from **speaker identification** (which of N known people is speaking) and different again from **speech recognition** (what words are being said). Speaker verification only cares about *voice identity*, not content.

The way this is done in modern deep learning systems:

1. Feed the audio into a neural network.
2. The network outputs a fixed-length numeric vector (e.g., 192 numbers) called a **speaker embedding**. Think of it as a "voice fingerprint" — a point in a high-dimensional mathematical space where recordings from the same person land close together and recordings from different people land far apart.
3. To verify a claim, extract the embedding of the enrolled (reference) voice and the embedding of the test voice, then measure the distance/similarity between them (usually **cosine similarity** — the cosine of the angle between the two vectors; 1 means identical direction, 0 means unrelated, -1 means opposite).
4. If the similarity is above a threshold, accept the claim; otherwise reject it.

The entire ECAPA-TDNN paper is about designing a **better neural network architecture** to produce these embeddings, so that same-speaker vectors cluster tighter and different-speaker vectors separate further.

## 1.2 Background concepts explained from scratch

### a) TDNN (Time Delay Neural Network)
A TDNN is basically a 1-dimensional convolutional neural network applied along the *time* axis of speech features. Each "frame" of audio (a short ~25 ms slice, taken every ~10 ms) is represented by a feature vector (see MFCC below). A TDNN layer looks at a small window of consecutive frames (its "context") and combines them — similar to how a CNN's kernel slides over an image, but here it slides over time. Stacking several such layers lets the network build progressively wider "receptive fields" in time, so deeper layers see more temporal context (i.e., they can "hear" longer stretches of speech at once).

### b) MFCC (Mel-Frequency Cepstral Coefficients)
Raw audio waveforms are hard for a network to use directly at low compute cost, so audio is first converted into a compact frequency-domain representation. MFCCs approximate how the human ear perceives sound: the frequency axis is warped onto the **Mel scale** (which is denser at low frequencies, matching human hearing), the signal is passed through a bank of filters, log-compressed, and then a compression transform (Discrete Cosine Transform) decorrelates the values. The result, per 25 ms frame, is a small vector (in this paper, **80 numbers**) that describes the "shape" of the spectrum at that instant — useful for both speech and speaker information. This paper uses 80-dimensional MFCCs computed every 10 ms.

### c) x-vector
"x-vector" is the historical name for the first widely-used TDNN-based speaker embedding system (Snyder et al., 2018). Its recipe:
- Frame-level TDNN layers process each time step, sharing weights across time (like a 1-D CNN).
- A **statistics pooling** layer collapses the variable-length sequence of frame vectors into one fixed-length vector by computing the **mean** and **standard deviation** across time of each channel/feature. This is the key trick that lets a network handle utterances of different lengths (2 seconds or 20 seconds) and still emit a fixed-size embedding.
- Two fully-connected layers follow; the first one is the **bottleneck** — a narrow layer (128–256 dimensions) whose output *is* the speaker embedding, because it forces the network to compress everything speaker-relevant into a small vector.
- The whole network is trained as a **speaker classifier** (i.e., "which of these N training speakers said this?") using a softmax-based loss. After training, the network is not used to classify — the classification head is thrown away, and the bottleneck-layer activations become the embedding.

### d) Extended-TDNN (E-TDNN) — Baseline 1
An improved x-vector variant that interleaves dilated 1-D convolution frame layers with plain dense layers and adds **residual (skip) connections** between frame layers, plus an **attentive statistics pooling** layer (see below) instead of plain mean/std pooling.

### e) ResNet and r-vector — Baseline 2
**ResNet** (Residual Network, He et al. 2016) is a landmark computer-vision architecture. Its core idea: instead of forcing a stack of layers to learn a full transformation `H(x)`, let them learn only the *residual* `F(x) = H(x) - x`, then add the input back: `output = F(x) + x`. This "skip connection" lets gradients flow directly backward through the addition during backpropagation, which fixes the **vanishing gradient problem** (gradients shrinking to near-zero as they pass through many layers, which stalls training of deep networks) and lets much deeper networks train successfully. **r-vector** applies ResNet18/ResNet34 (originally designed for 2-D images) to speech by treating the spectrogram as a 2-D image (time × frequency) instead of a 1-D sequence.

### f) Residual / skip connections
A shortcut path that lets information (and gradient) bypass one or more layers and be added back later. Benefits: faster convergence, avoids vanishing gradients, and lets the network default to "do nothing extra" (identity mapping) if a block isn't useful, making very deep networks safe to train.

### g) Statistics pooling and Attentive Statistics Pooling
Plain statistics pooling treats every time frame as equally important when computing the mean/std. But some frames (e.g., silence, noise, or unclear speech) carry less useful speaker information than others. **Attentive statistics pooling** (Okabe et al., 2018) adds a small neural attention mechanism that learns to assign a weight `α_t` (softmax-normalized so all weights across time sum to 1) to every time frame, then computes a *weighted* mean and weighted standard deviation instead of a plain average. This effectively lets the network do a soft form of Voice Activity Detection (VAD) — automatically down-weighting frames it doesn't find useful — without needing an explicit separate VAD step.

### h) Attention mechanism (general concept)
An attention mechanism is a small trainable sub-network that computes an "importance score" for each element of a set (here, each time frame), normalizes those scores into weights (usually via **softmax**, which turns arbitrary numbers into positive weights that sum to 1, emphasizing the largest values), and then uses those weights to combine the elements — a weighted average instead of a plain average. This lets the network dynamically focus more computation/attention on the parts of the input that matter most for the task.

### i) Squeeze-and-Excitation (SE) blocks
Borrowed from computer vision (Hu et al., 2018, "Squeeze-and-Excitation Networks"). The idea: not all channels (feature maps) in a convolutional layer are equally useful for every input; some channels should be amplified, others suppressed, depending on the *global* content of that particular input. An SE block does this in two steps:
- **Squeeze**: compute one summary number per channel by averaging that channel's values across the whole time (or spatial) dimension — this "squeezes" the map down to a per-channel descriptor that captures a global view of the utterance.
- **Excitation**: pass that per-channel descriptor through a tiny bottleneck neural network (a dimensionality-reduction layer, a non-linearity, then an expansion layer back to the original number of channels, ending in a **sigmoid** function that outputs values between 0 and 1 for each channel). These are re-scaling weights.
- The original feature map is then multiplied channel-wise by these learned weights, i.e., channels the network decides are important for this specific input get amplified, others get muted.

Why it matters for speech: a TDNN frame layer's convolution only "sees" a small window of nearby frames (limited by kernel size/dilation), so it has no idea what's happening in the rest of the recording. The SE block injects **global, utterance-level context** (background noise level, recording channel, overall pitch, etc.) into every local computation, letting local frame decisions be informed by the whole recording.

### j) Res2Net module
A refinement of the standard ResNet "bottleneck" convolution block (Gao et al., 2019). Instead of one single convolution processing all channels together, the input channels are split into several smaller groups; convolutions are applied to each group in a hierarchical, cascading fashion (each group's convolution output is added to the next group's input before its own convolution), which lets the layer represent features at **multiple scales** (multiple effective receptive field sizes) using far fewer parameters than one giant convolution would need, because the cascading effectively builds up a range of receptive fields for free.

### k) SE-Res2Block (the paper's core building block, Figure 1)
The paper fuses all of the above into one repeated unit:
1. A 1×1 (context-of-1-frame) dense/convolution layer that reduces the channel dimensionality (a cheap "squeeze" of the width, unrelated to the SE-squeeze, just a bottleneck).
2. A **Res2Net-style dilated 1-D convolution** (kernel size `k`, dilation `d`, scale dimension `s = 8` meaning the channels are internally split into 8 groups) that builds temporal context.
3. Another 1×1 layer restoring the original number of channels.
4. An **SE-block** that rescales every channel using global utterance statistics.
5. A residual/skip connection adding the block's input back to its output.

This single block is the repeating "Lego brick" of the whole network (used three times in the main architecture, at dilation spacings 2, 3, and 4 — meaning each successive block sees a progressively wider temporal context).

### l) Multi-layer Feature Aggregation (MFA)
A plain x-vector only pools the *last* frame layer's output. But different depths of a deep network encode different levels of abstraction (shallow layers: more acoustic/local detail; deep layers: more abstract, identity-related detail). The paper instead **concatenates the outputs of all three SE-Res2Blocks** (channel-wise, at every time step) before pooling, so that both shallow and deep information contribute to the final speaker statistics. A dense layer (1×1 convolution, `1536 × T` in the figure, since 3 blocks × 512 channels = 1536) mixes this concatenated information before it enters the pooling layer.

### m) Summed residual connections between blocks
On top of the individual SE-Res2Block's own internal skip connection, the paper also connects the blocks to each other: each SE-Res2Block's residual input is defined as the **sum of the outputs of every previous block** (not just the immediately preceding one), rather than a concatenation. Summation (rather than concatenation) is used specifically to keep the parameter count from growing, since concatenation would keep increasing channel width at every block.

### n) Channel- and context-dependent attentive statistics pooling (the paper's main pooling contribution)
Standard attentive pooling computes one attention weight per **time frame** that is shared across all channels — i.e., the whole feature vector at time `t` gets one importance score. The paper's insight: different channels (which loosely correspond to different learned "speaker-relevant properties," e.g., some channels might capture vowel-related characteristics, others consonant-related characteristics) may be informative at *different* times. So instead of one attention score per frame, the network computes **one attention score per (frame, channel) pair**: `e_{t,c}`.

Mathematically (as in the paper):
- `e_{t,c} = v_c^T · f(W·h_t + b) + k_c` — project the frame's activation vector `h_t` through a small shared linear layer (`W`, `b`) into a lower-dimensional space, apply a non-linearity `f` (e.g., tanh/ReLU), then combine with a channel-specific weight vector `v_c` and bias `k_c` to get a raw score for channel `c` at time `t`.
- `α_{t,c} = softmax over t of e_{t,c}` — normalize scores across *time* separately for each channel, so for every channel the attention weights across all frames sum to 1.
- Weighted mean per channel: `μ̃_c = Σ_t α_{t,c} · h_{t,c}`
- Weighted standard deviation per channel: `σ̃_c = sqrt( Σ_t α_{t,c} · h_{t,c}^2 − μ̃_c^2 )`
- The final pooled vector is the concatenation of the weighted mean vector and the weighted standard-deviation vector (so if there are 1536 channels, pooling outputs 3072 numbers, exactly as shown in Figure 2).

**Global context extension**: before computing the attention score, the paper also concatenates each frame's local activations `h_t` with the plain (unweighted) global mean and standard deviation of the whole utterance, so the attention mechanism can "see" overall recording properties (background noise level, channel/recording condition) when deciding how much to trust each frame — the same global-context philosophy as the SE-blocks, applied here to the pooling stage instead of the frame layers.

### o) Batch Normalization (BN)
A layer that, during training, normalizes the activations flowing through the network (subtracting the batch mean, dividing by the batch standard deviation, then applying a small learnable rescale/shift) to keep the training numerically well-conditioned. This reduces *internal covariate shift* (activation distributions "drifting" as earlier layers update) and generally speeds up and stabilizes training of deep networks.

### p) The overall architecture (Figure 2), read top-to-bottom
1. Input: 80×T (80 MFCC channels, T time frames).
2. Conv1D + ReLU + BN (kernel=5, dilation=1) — initial context-building layer, produces C channels (C=512 or C=1024, the paper's two configurations).
3. SE-Res2Block (k=3, d=2) → SE-Res2Block (k=3, d=3) → SE-Res2Block (k=3, d=4), each C×T, with the summed-residual connections described above.
4. Concatenate the three block outputs → 3×(C×T), i.e., MFA.
5. Conv1D + ReLU (kernel=1) → compress to 1536×T.
6. Attentive Statistics Pooling + BN → 3072×1 (mean+std for 1536 channels).
7. Fully-Connected + BN → 192×1 — this is the final speaker embedding.
8. AAM-Softmax classification head — used only during training (see below), discarded at inference time.

### q) AAM-Softmax (Additive Angular Margin Softmax, "ArcFace" loss)
A modified version of the softmax classification loss, originally developed for face recognition (Deng et al., 2019, "ArcFace"), adapted here for speaker classification. Standard softmax just tries to get the correct class's score highest. AAM-Softmax instead operates on the **angle** between the embedding and each class's weight vector (after normalizing both to unit length) and adds an **angular margin penalty** to the correct class, forcing the network to push same-speaker embeddings not just "slightly closer" but *angularly much closer* to their class center and further from other classes than plain softmax would require. This produces embeddings with much better geometric separation between speakers — exactly the property needed for verification (where separation, not just classification accuracy, is what's evaluated). The paper uses a margin of 0.2 and a softmax "prescaling" of 30 (a temperature-like scale factor applied before the softmax to control how peaked the probability distribution is).

### r) Cyclical Learning Rate ("triangular2" policy)
Instead of a learning rate that only decreases over training, a cyclical schedule repeatedly ramps the learning rate up and back down between a low bound (1e-8) and a high bound (1e-3) over a fixed number of iterations (one "cycle" = 130,000 iterations here). "Triangular2" means each successive cycle's peak is halved. This has been shown empirically to help escape poor local minima early in training and fine-tune more precisely later.

### s) Adam optimizer
A widely used gradient-descent variant that keeps a per-parameter adaptive learning rate, using running estimates of both the gradient's mean (momentum) and its variance, generally converging faster and more robustly than plain stochastic gradient descent.

### t) Data augmentation used
Because deep networks need lots of varied data to generalize, the authors create 6 extra augmented copies of every training utterance:
- **MUSAN noise/babble** — mixing in recorded background noise or crowd chatter.
- **RIR (Room Impulse Response) reverberation** — convolving clean speech with recorded room acoustic responses to simulate different rooms/reverb.
- **SoX tempo perturbation** (speed up / slow down) — changes speaking rate without necessarily changing pitch as much as naive resampling.
- **FFmpeg codec compression** (opus/aac, alternating) — simulates the quality loss from real-world compressed audio (e.g., VoIP calls).
- **SpecAugment** — a masking-based augmentation applied directly on the log-mel spectrogram: randomly blanks out 0–5 contiguous time frames and 0–10 contiguous frequency channels, forcing the network to not over-rely on any single time/frequency region.

### u) Evaluation metrics: EER and MinDCF
- **EER (Equal Error Rate)**: as you sweep the accept/reject decision threshold, the **False Acceptance Rate** (wrongly accepting an impostor) and **False Rejection Rate** (wrongly rejecting the genuine speaker) trade off against each other. EER is the point where these two error rates are equal. Lower EER = better system.
- **MinDCF (minimum normalized Detection Cost Function)**: a cost-weighted metric that combines false-accept and false-reject rates at the *operating point that minimizes total cost*, given assumed costs and a prior probability of a genuine attempt (`P_target = 10⁻²` here, meaning genuine trials are assumed rare — realistic for security use-cases like fraud prevention, where most incoming calls are *not* an attack). This metric is more representative of real deployment than EER because it explicitly accounts for the fact that false accepts (letting an impostor through) and false rejects (annoying a genuine user) may have very different real-world costs.
- **PLDA** (Probabilistic Linear Discriminant Analysis) — mentioned as an alternative, more complex scoring backend to plain cosine similarity; it's a generative statistical model that explicitly separates "between-speaker" variability from "within-speaker/session" variability, and is often used as a scoring function on top of embeddings rather than cosine distance, though this paper primarily reports results using cosine distance with **adaptive s-norm** (a score-normalization technique using an "impostor cohort" of average embeddings to make similarity scores more comparable/calibrated across different trials).

## 1.3 Results, in plain terms

- On VoxCeleb1 test sets, the best ECAPA-TDNN configuration (C=1024, ~14.7M parameters) achieves **0.87% EER**, beating the strongest baseline (ResNet34, 23.9M parameters, 1.19% EER) while using **fewer than half the parameters**. The smaller ECAPA-TDNN (C=512, only 6.2M parameters) still beats every baseline including the much bigger ones.
- Averaged across test sets, the large ECAPA-TDNN gives an **18.7% relative improvement in EER** and **12.5% relative improvement in MinDCF** over the best baseline per test set.
- The ablation study (Table 2) isolates each contribution's impact on the smaller (C=512) model:
  - Removing the SE-blocks hurts the most among single-component removals (20.5% relative EER degradation) — global-context channel rescaling is the single biggest contributor.
  - Replacing Res2Net convolutions with plain convolutions costs 5.6% relative EER, but Res2Net's real payoff is a **30% reduction in parameters** for similar-or-better performance.
  - Removing Multi-layer Feature Aggregation costs 8.2% relative EER — shallow-layer information genuinely helps.
  - The channel-and-context-dependent attention pooling beats plain attentive pooling by 9.8% relative EER, and the added global-context vector contributes a further, smaller 1.9% relative gain.
  - Summed residual connections (vs. plain adjacent-block skip connections) give a further 6.5% relative EER gain, though very slightly worse MinDCF — the authors kept it because it also generalized well in an external evaluation (the 2020 SdSV Challenge).

## 1.4 Why this paper matters for anti-spoofing / voice-cloning defense (not just verification)

ECAPA-TDNN itself is **not** an anti-spoofing/deepfake detector — it was designed purely to tell *which human* is speaking, assuming the speech is real. However, its embeddings are extremely widely reused as a **speaker-identity feature extractor** inside larger anti-spoofing and voice-cloning-detection pipelines, because:
- A cloned/synthesized voice, even if perceptually convincing, often produces an ECAPA-TDNN embedding that drifts subtly away from the real enrolled speaker's embedding cluster, especially over a longer call — because current voice-cloning/TTS systems don't perfectly replicate every fine-grained spectral/prosodic identity cue that ECAPA-TDNN was trained to be sensitive to.
- It's therefore commonly combined with a dedicated **spoof/liveness classifier** (such as AASIST or RawNet2, trained directly on real-vs-fake data like ASVspoof) so that one branch answers "is this a real recording of *a* human?" and the other answers "does this recording's identity stay consistent with the enrolled/claimed speaker?" — combining both signals is exactly the fusion idea used in the SIH project described in Part 2.

---

# PART 2 — VoiceGuardAI / "HOMADOS AI" — Full Technical & Logistical Plan
### Smart India Hackathon 2026 · Problem Statement 26104 · Theme: Blockchain & Cybersecurity · Team: TapuSena

## 2.1 Problem being addressed

**PS 26104**: "AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks." Modern text-to-speech (TTS) and voice-conversion (VC) systems can synthesize a convincing clone of a target person's voice from a small amount of reference audio. Attackers use this to impersonate executives, bank customers, or relatives on live phone calls to authorize fraudulent fund transfers or extract sensitive information ("vishing" / voice-deepfake fraud). Existing academic anti-spoofing systems (e.g., AASIST, RawNet2) are built and evaluated on **single, short utterances** — they were never designed to reason over an entire, multi-minute live phone call.

## 2.2 Proposed solution, explained end-to-end

The system (internally named "HOMADOS AI", productized as **VoiceGuardAI**) is a **real-time, multi-signal, call-duration risk-scoring engine**, not a single yes/no classifier. It fuses three complementary signal types into a single continuously-updating "impersonation risk score" during a live call:

1. **Spectral/prosodic spoof-artifact detection** — an AASIST-style graph-attention network looking for statistical artifacts that TTS/VC vocoders leave behind (unnatural pitch micro-variation, phase artifacts, spectral discontinuities, unnatural formant transitions) that are very hard for current generative models to fully erase.
2. **Speaker-embedding identity-consistency checking** — using ECAPA-TDNN embeddings (Part 1) computed on short rolling windows of the call, to check that the *voice identity* stays consistent with the enrolled/claimed identity throughout the call, since some spoofing/replay attacks are stitched together from multiple source clips or drift over a long call.
3. **Call-context/streaming reasoning** — instead of scoring one clip in isolation, chunk-level scores from (1) and (2) are aggregated over the call's duration using a lightweight sequence-reasoning/graph-node structure, so a momentary suspicious chunk doesn't trigger a false alarm, but a sustained pattern does.

### Processing pipeline (5 stages, matching the deck)

**Stage 1 — Audio capture & preprocessing**
- Ingest live call audio (from a telecom/enterprise integration point, or an uploaded/streamed call feed for demo purposes).
- **Voice Activity Detection (VAD)** removes silence/non-speech segments so downstream models only process actual speech.
- **Chunking**: split the continuous stream into overlapping windows (e.g., 2–4 seconds, matching typical spoof-detector training window sizes) for streaming/near-real-time processing rather than waiting for the whole call to end.
- **Denoising**: basic spectral noise suppression to normalize input quality across different telephony channels (VoIP, PSTN, cellular codecs).

**Stage 2 — Feature extraction**
- Raw waveform features (fed directly to RawNet2's end-to-end convolutional front-end, which learns its own filterbank instead of using hand-crafted features).
- Spectral/prosodic features (log-mel spectrograms / MFCCs as in Part 1) for the AASIST graph-attention branch.
- Speaker-embedding features: 192-dimensional ECAPA-TDNN embedding per chunk (and one long-term "enrollment" embedding computed from an initial trusted segment or a pre-registered voice sample of the claimed speaker).

**Stage 3 — Fused graph-attention scoring**
- The **AASIST** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) branch models spoof artifacts jointly across the spectral and temporal axes using graph attention (treating time-frequency regions as graph nodes so the network can attend to relationships between spectral and temporal artifacts simultaneously, rather than only convolving locally). Output: an authenticity score per chunk (real vs. synthetic/converted).
- The **speaker-consistency** branch computes the cosine similarity between each chunk's ECAPA-TDNN embedding and the enrolled speaker's reference embedding. Output: an identity-consistency score per chunk.
- These two per-chunk scores are combined (the "fused graph-attention model" in the deck) into one authenticity-plus-identity score per chunk — this is the project's core technical innovation beyond the published research (single-utterance detectors don't do this fusion, and don't reason across chunks of one continuous call).

**Stage 4 — Real-time risk-scoring engine**
- Aggregates the stream of per-chunk fused scores across the call's duration (e.g., an exponentially-weighted running score, or a small sequence model / graph-node update rule) to produce one continuously updating **call-level impersonation risk score**, incorporating call context (e.g., is this a high-value transaction call, is the claimed identity a VIP/executive profile, has risk been trending upward over the last N chunks).

**Stage 5 — Threshold-based alerting/response**
- Below a low threshold: normal, no action.
- Mid threshold: soft UI warning shown to the human agent/system (e.g., "elevated risk — verify identity").
- High threshold: trigger escalation actions — require callback verification, step-up authentication, or block/hold a pending transaction/approval — before the flagged action (e.g., a fund transfer) is permitted to proceed.

## 2.3 Where the training and testing data will come from (detailed, dataset-by-dataset)

The project needs data for **two distinct sub-models**, which are trained/fine-tuned somewhat separately and then combined at inference time:

### (A) Speaker-embedding model (ECAPA-TDNN) — identity representation

We do **not** plan to train ECAPA-TDNN from scratch (this alone would need weeks of large-scale compute for limited marginal benefit over strong existing checkpoints). Instead we plan to use **SpeechBrain's publicly released, pre-trained ECAPA-TDNN checkpoint** (trained on VoxCeleb1+VoxCeleb2 by the SpeechBrain team, and freely licensed for research/hackathon use) as the identity-embedding backbone, and **fine-tune only the top layers** (or fine-tune fully, resources permitting) on our specific downstream task (identity-consistency over telephony-quality audio), rather than reproducing the original multi-GPU-week pretraining.

If/when we do fine-tune or partially retrain this backbone, the source data is:
- **VoxCeleb1**: 1,251 speakers, ~153,516 utterances, ~352 hours, YouTube "in the wild" interviews/speeches — publicly downloadable for research use from the University of Oxford VGG group (subject to their dataset terms and the fact that raw audio must currently be obtained via YouTube URLs/timestamps or an authorized mirror, since original direct hosting has been restricted; a common workaround used by many research groups is downloading from academic mirrors that redistribute the dataset under the original license terms, or via the `voxceleb_trainer`/`SpeechBrain` community download scripts).
- **VoxCeleb2**: 6,112 speakers (5,994 in the standard "dev" training split), ~1.09 million utterances, ~2,442 hours, same collection methodology (automated face-voice synchronization pipeline on YouTube videos) — this is the dataset the original ECAPA-TDNN paper itself trains on.
- Combined, these two datasets give **~145+ nationalities, near-gender-balance, a wide age range, and heavy real-world acoustic diversity** (interviews, red carpets, outdoor events, studio recordings, handheld-device recordings) — which is exactly the diversity needed so the identity model doesn't overfit to clean, quiet, single-accent speech.

### (B) Spoof/liveness detection model (RawNet2 + AASIST backbone)

This is the model that must be fine-tuned specifically for our task (distinguishing real human speech from TTS/VC-cloned speech), since this is not what a generic speaker-embedding model is trained to do.

- **ASVspoof 2019 — Logical Access (LA) partition**: built on the VCTK corpus (107 speakers total: 46 male, 61 female). Official split: **Train — 20 speakers, 2,580 bona-fide + 22,800 spoofed utterances (6 attack algorithms, A01–A06)**; **Dev — 20 speakers, 2,548 bona-fide + 22,296 spoofed utterances (same 6 attack algorithms)**; **Eval — 67 speakers, 7,355 bona-fide + 63,882 spoofed utterances, but generated with 13 different, unseen attack algorithms (A07–A19)**, deliberately disjoint from the attacks seen in Train/Dev, specifically to test generalization to unseen spoofing methods — precisely the property we need, since real attackers won't use the exact TTS/VC systems we trained on.
- **ASVspoof 2021 — LA and DF (DeepFake) partitions**: adds ~181,566 utterances in the LA condition alone, this time additionally passed through real telephony transmission channels (VoIP, PSTN codecs), which is highly relevant to us since our production use-case *is* phone calls; the DF partition specifically targets more modern neural deepfake generation and compression artifacts, closer to today's voice-cloning threat landscape than the original 2019 set.
- **AASIST's own public GitHub reference implementation** (`clovaai/aasist`) already provides training/scoring code and pretrained checkpoints on ASVspoof 2019 LA, which we plan to use as our starting checkpoint for fine-tuning rather than training this backbone completely from random initialization.

### (C) Our own supplementary/field dataset (the part that gets the 60-20-20 split)

Since the public spoof-detection datasets have their own fixed official train/dev/eval speaker-disjoint splits (which we will **keep intact**, because breaking them would let information leak across splits and inflate our reported numbers artificially), the place we apply a clean, self-defined **60% / 20% / 20% (train / validation / test)** split is our own **project-collected, call-context evaluation dataset**, which is necessary because none of the public datasets simulate a full, multi-minute, streaming phone call with a live risk score — they only provide single, isolated utterances. This dataset will combine:
- Volunteer-recorded genuine calls (own team members, with consent, across multiple devices/networks/rooms).
- Synthetic/cloned calls generated in-house using a small set of open-source TTS/voice-conversion tools (e.g., open-source neural TTS and VC toolkits) applied to volunteers' voices, to create realistic attack simulations for demo and stress-testing.
- Publicly available open call-center/telephony audio samples (non-sensitive, license-permitting) for background/channel realism.

Planned composition and scale for the hackathon prototype phase (explicitly labeled as an *estimate*, to be refined once data collection begins):
- Target: roughly **800–1,200 minutes (13–20 hours) of labeled call-style audio**, combining genuine and spoofed/cloned sessions, across at least **8–15 distinct volunteer speakers**, multiple accents/languages represented in the team and available volunteers, and at least 3 different simulated channel conditions (clean, VoIP-compressed, noisy background) to mirror real deployment variability.
- Split: **60% (≈480–720 min) for training/fine-tuning the fusion + risk-scoring layer, 20% (≈160–240 min) for validation/hyperparameter tuning and threshold calibration, 20% (≈160–240 min) held out purely for final testing/demo metrics**, split at the *speaker/session* level (not just randomly by clip) so that no speaker or call session appears in more than one split — this avoids the model "memorizing" a specific voice instead of learning general spoof/consistency cues.
- This self-collected set is deliberately small compared to VoxCeleb/ASVspoof, because its purpose is **fine-tuning the fusion + streaming risk-aggregation logic** (Stage 3–4 of the pipeline) on top of the already-pretrained backbones, not training a speaker or spoof model from zero — training those large backbones from scratch would require the full VoxCeleb2/ASVspoof scale (millions of utterances, thousands of hours) which is neither necessary nor feasible within a hackathon timeline.

### Training-time estimate

- Fine-tuning the AASIST/RawNet2 spoof backbone on ASVspoof 2019 LA (Train: ~25,380 utterances) for a modest number of epochs (10–20) on a single modern GPU (e.g., an NVIDIA RTX 3090/4090-class or a shared A100 partition) is estimated at roughly **4–10 GPU-hours**, based on the scale of the official dataset and typical published fine-tuning times for these architectures.
- Fine-tuning/adapting the ECAPA-TDNN top layers on our supplementary dataset is a much smaller job (tens of thousands of short clips at most) — estimated at **1–3 GPU-hours**.
- Training the fusion/risk-aggregation layer (a small model sitting on top of two already-trained backbones, operating on chunk-level score sequences rather than raw audio) is lightweight — estimated at **under 1 GPU-hour**.
- **Total estimated compute for the hackathon prototype: roughly 6–15 GPU-hours**, which is realistically achievable on a single good GPU over 1–2 days, or in a few hours on a shared multi-GPU node — this is *not* a "train ECAPA-TDNN or AASIST from scratch on VoxCeleb2" job (which the original papers report taking multiple days across multiple GPUs); we are deliberately building on public pretrained checkpoints to keep this within hackathon-realistic compute and time budgets.

### Supercomputer / high-performance compute access plan

Since AICTE is running SIH 2026 and typically provides or facilitates access to national compute resources for shortlisted teams, our plan (in order of preference) is:
1. **Apply through AICTE/SIH's official channel for compute credits or access to the National Supercomputing Mission (NSM) PARAM series facilities** (e.g., PARAM Shivay/Param Siddhi-class clusters made available to student teams via participating institutions), which several SIH cohorts have been granted access to in past editions for GPU-accelerated training.
2. **Institutional HPC/GPU lab access** through our college's computer science department or a partner AICTE-recognized institution, if available, as a fallback/supplement.
3. **Cloud GPU credits** (e.g., through free/education tiers or startup/hackathon credit programs commonly offered by major cloud providers, or through Google Colab Pro / Kaggle's free GPU quotas for early prototyping) to cover incremental experimentation beyond what local/institutional hardware can provide.
4. Given the modest total compute estimate above (6–15 GPU-hours for the actual model work, plus additional hours for experimentation/ablation), a single shared high-end GPU node is realistically sufficient for the entire prototype; true supercomputer-scale access would mainly be requested to allow **parallel experimentation** (trying multiple fusion strategies, hyperparameters, and augmentation configurations simultaneously) rather than because any single training run is computationally unreachable otherwise.

*(Note: exact supercomputer allocation is something to explicitly ask the SIH nodal center/AICTE about at the evaluation stage — since actual availability and application process for any given edition should be confirmed with the organizers rather than assumed, this document states our access strategy but the final allocation is outside our team's direct control.)*

## 2.4 Technology stack — what, where from, and why (with technical justification)

| Layer | Technology | Source | Why chosen |
|---|---|---|---|
| Core ML framework | **Python + PyTorch** | Open-source (PyTorch Foundation) | Industry-standard for speech/deep-learning research; PyTorch's dynamic computation graph and huge ecosystem of pretrained speech models makes it the natural fit, and both SpeechBrain and the reference AASIST/RawNet2 codebases are PyTorch-native, so no framework-conversion overhead. |
| Audio processing | **librosa, torchaudio** | Open-source | `librosa` provides mature, well-tested implementations of MFCC/mel-spectrogram extraction, resampling, and audio I/O for preprocessing and experimentation; `torchaudio` provides GPU-accelerated, PyTorch-native equivalents so feature extraction can run as part of the training/inference graph without CPU bottlenecks — important for real-time inference. |
| Speaker embeddings | **SpeechBrain (ECAPA-TDNN)** | Open-source toolkit, pretrained checkpoints from the SpeechBrain team (trained on VoxCeleb1+2) | Gives us a production-quality, already-validated ECAPA-TDNN implementation and checkpoint (Part 1's architecture) instead of re-implementing and re-training the whole paper from scratch — lets the team focus engineering effort on the *fusion and streaming logic*, which is the actual innovation of this project. |
| Spoof/liveness detection | **RawNet2 + AASIST**, fine-tuned on **ASVspoof** | Original authors' open-source repos (`clovaai/aasist`, RawNet2 reference implementation), fine-tuned by us on ASVspoof 2019/2021 | These are the current strongest published open architectures specifically for the spoof-detection sub-task (as opposed to ECAPA-TDNN, which is an identity model, not a spoof detector) — RawNet2's raw-waveform front-end captures artifacts that hand-crafted spectral features can miss, while AASIST's graph attention over both spectral and temporal axes captures cross-domain spoofing cues; combining both gives complementary detection signals. |
| Backend/serving | **FastAPI + WebSockets + Redis** | Open-source | FastAPI gives a modern, async-first Python API framework suited to low-latency inference serving; WebSockets are required (rather than plain REST) because the system needs a **persistent, bidirectional streaming connection** to push updated risk scores continuously during a live call rather than one-shot request/response; Redis provides fast in-memory storage for **live session state** (the running per-call risk aggregation, chunk history, enrolled-speaker embedding cache) that must be read/written with very low latency during a live call. |
| Frontend/dashboard | **React + Chart.js** | Open-source | React for a component-based, responsive operator dashboard; Chart.js for lightweight, real-time-updating visualizations (the live risk gauge and waveform display) without the overhead of heavier charting/visualization libraries — appropriate given the dashboard's relatively simple, live-metric-focused visual needs. |
| Integration | **REST/gRPC APIs, Docker** | Open standards / open-source | REST for simple external integrations (e.g., a bank's fraud system polling risk status); gRPC as a lower-latency, strongly-typed option for high-throughput telecom/enterprise integrations where per-call latency matters; Docker containerization so the whole stack (models + backend + dependencies) can be deployed consistently across a bank's, enterprise's, or telecom operator's infrastructure regardless of their underlying environment. |

## 2.5 Why the 60-20-20 split is applied where it is (recap of the key nuance for evaluators)

It's important to be able to explain clearly in the demo/evaluation that **the public benchmark datasets (VoxCeleb, ASVspoof) already come with fixed, speaker-disjoint official splits that the community uses for fair comparison**, and we deliberately preserve those splits rather than reshuffling them into 60-20-20, because changing official splits would make our reported EER/MinDCF numbers non-comparable to the published baselines in Part 1's paper and to other ASVspoof-based work. The 60-20-20 split is the one **we control and define ourselves** — applied to our own supplementary, project-collected call-simulation dataset, which is the piece of data that doesn't come with any pre-existing official split, and is what we use to train, validate, and finally test the **fusion + streaming risk-scoring layer** that is our actual novel contribution on top of the existing published research.

## 2.6 Summary — the technical narrative to present

1. Speaker verification 101 → the x-vector idea → ECAPA-TDNN's three innovations (SE-Res2Blocks for global-context channel attention, Res2Net for efficient multi-scale temporal modeling, and channel-and-context-dependent attentive statistics pooling) → why it beats deeper ResNet baselines with fewer parameters.
2. Where ECAPA-TDNN's embeddings fit into an anti-spoofing pipeline: not a spoof detector by itself, but a strong identity-consistency signal to fuse with a dedicated spoof classifier.
3. Our system's innovation: extending single-utterance spoof detection (AASIST/RawNet2) plus identity verification (ECAPA-TDNN) into a **streaming, call-duration risk-reasoning system**, which current published research does not directly address.
4. Data strategy: reuse large, standard, speaker-disjoint public datasets (VoxCeleb1+2 for identity, ASVspoof 2019+2021 for spoof detection) for the two pretrained backbones, and build/apply our own 60-20-20-split dataset specifically for training and evaluating the fusion and streaming risk-scoring logic — the genuinely new part of the system.
5. Compute strategy: fine-tune (don't retrain from scratch) on modest, realistically obtainable GPU compute (estimated 6–15 GPU-hours total), with a plan to request AICTE/NSM PARAM supercomputing access mainly to parallelize experimentation rather than because any single run is otherwise infeasible.
