# Blur Cull: training the model

Blur Cull (Photo ▸ Blur Cull… in LightCraft) scores how blurred each photo's **subject** is, from 0 (tack sharp) to
100 (unrecognisable). You pick a window on a histogram and LightCraft flags the photos inside it one way (Pick) and
the rest another (Reject). A window from 0 to 20 is a sharp-only cull. Widen it to keep intentionally soft or
artistic shots as well.

The scores come from **BlurNet**, a small network (150 k parameters, about a second per photo on a laptop CPU)
that you train here on **your own photos**. LightCraft ships without a model. This folder holds everything needed
to make one:

| File | What it does |
|---|---|
| `prepare.py` | reads your photo folders into a training cache (raws through `lightcraft-cli`) |
| `train.py` | trains BlurNet (GPU if PyTorch sees one, else CPU) |
| `export.py` | writes `blurnet.safetensors`, the file LightCraft installs |
| `score.py` | scores any folder outside LightCraft, with optional blur-map pictures, to check a model |
| `net.py`, `subject.py`, `synth.py`, `kernels.py` | the network, subject scoring, synthetic blur, and the blur measure. `crates/blur` is their Rust twin |
| `parity_reference.py` | reference outputs proving the Rust network matches PyTorch |

## How it learns

Every 16 × 16-pixel cell of a photo gets a blur value: the equivalent blur sigma in pixels, measured at the
analysis size (long edge 1600). On a log scale, that becomes the 0–100 score. The network learns it from two
kinds of examples.

1. **Your sharp photos, blurred by the trainer.** Each one becomes thousands of examples. The trainer applies a
   known amount and kind of blur:
   - defocus (lens discs, including cat's-eye bokeh)
   - hand shake (random jittery trajectories)
   - ghosting: 2–5 crisp copies shifted along a direction (shake, vibrating bodywork, stabilisation fighting a
     pan). Each copy keeps crisp edges, so without these, ghosted lettering reads as sharp
   - motion streaks
   - soft-lens Gaussian blur
   - partial blur through a random depth mask, like shallow depth of field
   - motorsport motion (about a third of examples): a car-shaped region tracked sharp or a little soft against a
     background streaked for 25–500 px (a pan), a missed pan where the car streaks too, or a still camera with the
     car streaking past. Long streaks keep crisp edges across their length, so without these a panned background
     reads as sharp and the "sharpest part of the frame" lands on it instead of the car.

   It then adds sensor noise, in-camera sharpening and JPEG compression. Because the trainer did the blurring,
   every cell's label is exact. This is where most of the learning happens, so **variety matters more than
   count**: people, landscapes, wildlife, night scenes, high and low ISO, all your cameras and lenses.
2. **Your real photos, rated by eye.** Put each one in a folder named after the score range it belongs in. The
   trainer pushes the photo's subject score into that range. This ties the scale to *your* judgement and teaches
   real-world blur the simulator misses (subject motion, missed AF, heavy noise reduction).

Where is the subject? LightCraft uses the camera's **focus point** when the file records one, then **face
regions**, and otherwise **the sharpest detailed part of the frame**. If anything in the frame is in focus, it is
almost always what you meant. A sharp bird on a creamy background therefore scores sharp, while a missed-focus or
shaken frame has no sharp part and scores blurry.

## 1. Set up (once)

You need Python 3.10 or newer.

```powershell
cd "E:\ArtCraft Repos\lightcraft"
python -m venv .venv-blurcull
.venv-blurcull\Scripts\activate            # macOS / Linux: source .venv-blurcull/bin/activate
# NVIDIA GPU: first install the CUDA build of PyTorch with the command from https://pytorch.org/get-started/locally/
pip install -r tools\blurcull\requirements.txt
cargo build --release -p lightcraft-cli    # reads raw files (ARW, CR3, NEF, RAF, DNG, …) for training
```

`.venv-blurcull/` is local only. Don't commit it.

## 2. Lay out your photos

Make a data folder **outside the repo** (photos are never committed), for example `D:\blurcull-data`:

```
D:\blurcull-data\
  sharp\                    tack-sharp photos: subject AND most detail crisp at 100 %
    birds\  portraits\  landscapes\  …      (subfolders are fine; names don't matter)
  rated\                    real photos you judged; the folder name ends in the score range
    sharp_0-12\             critically sharp
    slightly-soft_10-25\    fine on screen, soft at 100 %
    soft_22-40\             visibly soft subject
    blurry_38-65\           missed focus, shake
    very-blurry_62-100\     unusable, or deliberately abstract
    artistic_30-80\         intentional blur you like (panning, ICM): rated by how blurry, not by taste
```

- **`sharp\`** is the most important folder. Aim for **300+ photos** (1,000+ is better), any format LightCraft
  opens. Pick shots where nearly everything that has detail is in focus: landscapes at f/8, and portraits only if
  the background has little texture. `prepare.py` measures each one and sets aside any that look soft, so
  mistakes cost little.
- **`rated\`** helps a lot even when small: **20–100 photos per range** is plenty. The ranges may overlap, and the
  folder name only has to end in `_lo-hi`. Rate the subject, not the background.
- **Optional:** if the subject isn't the sharpest part of the frame (a deliberately soft subject in front of a
  sharp background), add `subjects.csv` to its rated folder, one line per photo: `file,x0,y0,x1,y1`, with
  coordinates from 0 to 1 (`IMG_0042.ARW,0.40,0.30,0.62,0.71`).

A good way to fill `rated\`: in LightCraft, flag or rate a few hundred photos, filter by rating, export them as
JPEG, then drop each group in its folder.

## 3. Prepare

```powershell
cd tools\blurcull
python prepare.py --data D:\blurcull-data
```

This writes `D:\blurcull-data\cache\` (analysis-size PNGs plus `manifest.json`) and lists:
- `sharp\` photos that look soft: they are skipped. Move them to `rated\`.
- files it couldn't read.

Run it again after adding photos. Only new ones are processed.

## 4. Train

```powershell
python train.py --data D:\blurcull-data
```

- The defaults are 30,000 steps of 64 crops. On a recent NVIDIA GPU that takes about 1–2 hours. On a CPU it takes
  much longer (the log prints an estimate). On a CPU, try `--steps 8000 --batch 32` first to see the shape of
  things.
- It prints validation every 1,000 steps:
  - `synthetic_mae_log2`: the error on held-out synthetic blur. Lower is better; below 0.3 is good.
  - `rated_in_range`: the share of held-out rated photos that land inside their folder's range. Higher is better.

  The best checkpoint is kept as `runs\best.pt`.
- Stopped it? `python train.py --data D:\blurcull-data --resume` carries on.
- `--rated-weight` (default 0.5) sets how strongly your ratings pull the scale. Raise it if your rated photos keep
  missing their ranges.

## 5. Export and check

```powershell
python export.py --data D:\blurcull-data --note "my photos, Oct 2026"
python score.py --model D:\blurcull-data\blurnet.safetensors D:\blurcull-data\rated --check --maps D:\blurcull-maps
```

`score.py --check` reports how many rated photos land in their range. `--maps` writes each photo with its blur map
painted on (green = sharp, red = blurry, untinted = nothing to judge), which is the quickest way to see what the
model thinks.

## 6. Install in LightCraft

Photo ▸ **Blur Cull…** ▸ **Install Model…** and pick `blurnet.safetensors`. The model is copied to
`<settings>\models\blurnet.safetensors` (`$LIGHTCRAFT_BLUR_MODEL` overrides the path). From a script, the same
step is `{"method": "engine.execute", "params": {"command": "blur.model.install", "params": {"path": "D:/blurcull-data/blurnet.safetensors"}}}`.

Then, in the dialog:
1. Choose **Selected** or **All in View** (browse a folder to cull that folder).
2. Click **Score N Photos**. Scoring runs in the background, with a row in the activity stack (✕ stops it).
3. Drag the window's edges on the histogram. Inside → Pick / Unflag / Leave; outside → Reject / Unflag / Leave.
4. Click **Apply Flags**. It is one undo step. Photos with no score are left alone.

Scores are kept beside the library in `blur-scores.json`. They are reused until the photo's file or the model
changes. Installing a retrained model makes the next scan score everything again.

## Retraining

Add photos and run `prepare.py`, `train.py` and `export.py` again, then install the new file. Training starts
fresh unless you pass `--resume`. Keep old exports if you want to compare them with `score.py`.

## Changing the network

`net.py` (`ARCH`) and `crates/blur/src/net.rs` (`LAYERS`) must stay identical, as must `subject.py` and
`crates/blur/src/subject.rs`. A new architecture gets a new name (`blurnet-v2`); the app refuses files whose
architecture it doesn't run. Check the Rust side against PyTorch with:

```
python parity_reference.py --model blurnet.safetensors --image some.jpg --out parity
LIGHTCRAFT_BLUR_PARITY=parity cargo test -p lightcraft-blur --test parity
```
