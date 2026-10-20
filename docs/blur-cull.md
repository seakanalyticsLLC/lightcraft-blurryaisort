# Blur Cull

Photo ▸ **Blur Cull…** scores how blurred each photo's **subject** is, from 0 (tack sharp) to 100
(unrecognisable). You choose a window on the scores' histogram, and LightCraft flags the photos inside it one way
and the rest another, in one undo step:

- **Sharp only:** window 0–20, inside → Pick, outside → Reject.
- **Keep intentional blur too:** window 0–75 (panning, ICM, dreamy portraits), outside → Reject. Only true misses
  go.
- **Find the artistic ones:** window 40–100, inside → Pick, outside → Leave.

It is a learned, subject-aware companion to **Assisted Culling** (Photo ▸ Assisted Culling…). Assisted Culling's
focus score is a classical measure of the sharpest area at thumbnail size. Blur Cull uses a model trained on your
own photos, at a larger analysis size (long edge 1600). It also knows where the subject is.

## By shutter speed

A slow pan is never as crisp as a 1/1000 frame, so one window for every photo either culls good pans or keeps soft
fast frames. Tick **By Shutter Speed** and each range of shutter speeds gets its own window. The defaults:

| Shutter speed | Keep | |
|---|---|---|
| 1/500 and faster | 0–15 | should be tack sharp |
| 1/500 – 1/60 | 0–25 | a little give |
| 1/60 – 1/15 | 0–40 | pans, handheld low light |
| Slower than 1/15 | 0–60 | slow pans, intentional camera movement |
| Unknown shutter speed | the dialog's own window | scans, exports without metadata |

- Each bracket runs from just slower than the bracket above it down to its limit, inclusive: a 1/500 frame is in
  "1/500 and faster", a 1/400 frame in "1/500 – 1/60".
- Click a row's button to put that bracket's photos on the histogram; the handles, From/To and the preset buttons
  then set that bracket's window. Each row shows how many of its photos fall inside and outside.
- Pick a bracket's limit from its menu (third stops, limited to the speeds between its neighbours). **Add Bracket**
  adds one a stop slower than the last limit; **−** removes one, and the bracket below takes its speeds. The last
  bracket always takes every slower speed. Up to 8 brackets.
- The brackets are saved with the app's settings (`ui.json`) and come back the next time the dialog opens.
- The shutter speed is the one in the photo's metadata (the Shutter Speed field in Metadata). The model doesn't see
  it: the score measures blur the same way for every photo, and the brackets only move where the line is drawn.

## Where the subject is

1. The camera's **focus point**, when the file records one ([focus-points.md](focus-points.md)).
2. Otherwise **face and pet regions** (XMP, or found by Settings ▸ Faces).
3. Otherwise **the sharpest detailed part of the frame**. If anything is in focus, that is what you meant, so a
   sharp subject on a soft background scores sharp. A missed or shaken shot has no sharp part and scores blurry.

Photos with nothing to judge (fog, a blank wall) get no score and are never flagged.

## The model

**The model is optional and not part of LightCraft.** It is trained on your own photos with the tools in
[`tools/blurcull`](../tools/blurcull/README.md): put tack-sharp photos in one folder (the trainer blurs them in
known ways, so every cell has an exact label) and a few rated real-world examples in folders named by score range.
Train, export `blurnet.safetensors`, then click **Install Model…** in the dialog (`blur.model.install`). The file
goes to `<settings>/models/blurnet.safetensors` (`$LIGHTCRAFT_BLUR_MODEL` overrides it).

The network is BlurNet v1: nine 3 × 3 convolutions, about 150 k parameters, run in pure Rust on candle
(`crates/blur`), on the CPU, at about a second per photo. A model file is checked before it is used (format,
architecture, every tensor's shape, finite values, size cap). A wrong or damaged file is refused with a message,
never a crash.

## Scanning and scores

- **Score N Photos** scores the photos that have no usable score yet, on a background worker. Its progress shows
  in the activity stack (kind `blur`), where ✕ stops it. The scan pauses while you drag, type or scroll.
- Each photo is read from a raw's embedded preview when that is at least 90 % of the analysis size. Otherwise it
  is rendered with default settings, upright and uncropped, the frame focus points and faces are given in.
- Scores are kept beside the library in `blur-scores.json`, not in the catalog, so a catalog opened by a build
  without Blur Cull is unchanged. A score is used while the photo's content and the installed model are the ones
  that made it. Installing a retrained model makes the next scan score everything again.

## Commands

| Command | |
|---|---|
| `blur.model.status` | installed?, model info, score count, scan progress |
| `blur.model.install {path}` / `blur.model.remove` | install (after checking) or delete the model file |
| `blur.scan {ids? \| scope?: selection \| view, rescan?, wait?}` | score in the background; `wait: true` for scripts |
| `blur.pump {pace?}` | called by the app every frame; collects results |
| `blur.scores {ids? \| scope?, photos?: bool, min?, max?, brackets?}` | scores, a 20-bin histogram, counts inside / outside a window; with `brackets`, the same for each bracket and for `unknownShutter` |
| `blur.apply {min, max, brackets?, inRange?, outOfRange?, ids? \| scope?}` | flag (`pick` / `reject` / `none` / `keep`), one undo step; with `brackets`, counts `byBracket` and `unknownShutter` |
| `dialog.blurCull {scope?, min?, max?, byShutter?}` (UI) | opens the dialog |

`brackets` is a list, fastest first, of `{upTo, min, max}`. `upTo` is the bracket's slowest shutter speed, as the
camera writes it (`"1/60"`, `"2"`) or in seconds. Only the last bracket may leave it out, to take every slower
speed. With brackets, `min`..`max` is the window for photos without a known shutter speed (or slower than every
bracket).

A script culling a folder through the control channel:

```json
{"id":1,"method":"engine.execute","params":{"command":"blur.scan","params":{"scope":"view","wait":true}}}
{"id":2,"method":"engine.execute","params":{"command":"blur.apply","params":{"scope":"view","min":0,"max":20,"inRange":"pick","outOfRange":"reject"}}}
```

The same cull by shutter speed:

```json
{"id":3,"method":"engine.execute","params":{"command":"blur.apply","params":{"scope":"view","min":0,"max":25,"brackets":[{"upTo":"1/500","min":0,"max":15},{"upTo":"1/60","min":0,"max":25},{"upTo":"1/15","min":0,"max":40},{"min":0,"max":60}]}}}
```
