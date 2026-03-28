# React Native Integration Guide for Chatterbox-Fastest

## Purpose

This guide explains how a React Native app can call the `chatterbox-fastest` API server, generate speech, receive the WAV response, save it locally, and use it for playback or other app workflows.

This document is based on the current server implementation in [tts_api_server.py](/chatterbox-fastest/tts_api_server.py).

## Current API Overview

The server currently exposes these endpoints:

- `GET /healthz`
- `GET /v1/languages`
- `POST /v1/tts`

The main endpoint your React Native app will use is:

- `POST /v1/tts`

Important: `/v1/tts` currently returns a WAV file, not MP3 and not MP4.

## What `/v1/tts` Returns

The response body is raw WAV audio bytes.

Response details:
- `Content-Type: audio/wav`
- body: binary WAV file data

Response headers:
- `X-Conditioning-Seconds`: reference-audio conditioning time
- `X-Queue-Wait-Seconds`: time spent waiting to be included in a generation batch
- `X-Generation-Seconds`: model generation time in seconds
- `X-T3-Seconds`: time spent generating speech tokens
- `X-S3Gen-Seconds`: time spent converting speech tokens into waveform audio
- `X-Wav-Encode-Seconds`: WAV serialization time
- `X-End-To-End-Seconds`: total server-side request time
- `X-Audio-Seconds`: output audio duration in seconds
- `X-Realtime-Factor`: generated-audio-seconds divided by generation-seconds
- `X-Chunks`: number of prompt chunks used internally
- `X-Batch-Requests`: number of HTTP requests merged into the same generation batch
- `X-Batch-Prompts`: number of prompt chunks merged into the same generation batch

## Observed App Performance On RTX 3080 Ti

Real requests from a React Native client to the FastAPI server on an RTX 3080 Ti showed the following single-request behavior with varying text lengths:

| audio seconds | conditioning | T3 | S3Gen | generation | realtime factor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `3.04s` | `0.29s` | `1.50s` | `0.78s` | `2.28s` | `1.33x` |
| `3.00s` | `0.00s` | `1.43s` | `0.38s` | `1.81s` | `1.66x` |
| `4.24s` | `0.00s` | `2.07s` | `0.46s` | `2.53s` | `1.67x` |
| `5.76s` | `0.00s` | `2.69s` | `0.42s` | `3.11s` | `1.85x` |
| `5.00s` | `0.00s` | `2.42s` | `0.40s` | `2.82s` | `1.77x` |

Average across those `5` requests:

- output audio: `4.208s`
- generation time: `2.510s`
- `T3`: `2.022s`
- `S3Gen`: `0.488s`
- realtime factor: `1.658x`

Interpretation:

- For single-request app traffic, the server is already returning full WAV output faster than the length of the generated audio on an RTX 3080 Ti.
- The first request may pay a small conditioning/setup cost, but steady-state single-request performance is stronger than the synthetic burst-load tests suggest.

## Network Setup for React Native

If your React Native app runs on a physical phone:

- do not use `http://127.0.0.1:8000`
- do not use `http://localhost:8000`

Instead, use the local IP address of the machine running the server, for example:

- `http://192.168.1.23:8000`

Requirements:
- phone and server must be on the same network
- port `8000` must be reachable
- your device must be allowed to connect over HTTP if you are not using HTTPS

Examples:

- Android emulator often uses `10.0.2.2` for host access
- iOS simulator can often use `localhost`
- physical devices should use your machine's LAN IP

## Server Startup Example

Example:

```bash
cp chatterbox-server.env.example chatterbox-server.env
./easy_start.sh
```

Default server bind:

- host: `0.0.0.0`
- port: `8000`

That means the server is reachable from other devices on your local network if networking permits it.

## Endpoint Details

### `GET /healthz`

Use this to confirm the server is alive.

Response:

```json
{"status":"ok"}
```

Suggested app usage:
- run this during app startup or before enabling the TTS feature
- show a friendly "server unavailable" message if it fails

### `GET /v1/languages`

Returns the language map supported by the currently loaded model.

If the server is running in multilingual mode, the response is a JSON object like:

```json
{
  "en": "English",
  "fr": "French",
  "de": "German",
  "zh": "Chinese"
}
```

If the server is running in English-only mode, the response is:

```json
{
  "en": "English"
}
```

Suggested app usage:
- fetch once on app startup
- populate a language picker dynamically
- do not hardcode multilingual options if the server may run in English-only mode

### `POST /v1/tts`

This is the text-to-speech endpoint.

It accepts either:
- `application/x-www-form-urlencoded`
- `multipart/form-data`

Use `multipart/form-data` when uploading a reference audio file for voice cloning.

## `/v1/tts` Request Parameters

These fields are currently accepted by the server.

### Required

- `text: string`
  - required
  - must not be empty

### Optional

- `audio_prompt: file`
  - optional uploaded audio file
  - if provided, the model uses it for voice conditioning / cloning
  - if omitted, the model uses default conditionals

- `language_id: string`
  - default: `"en"`
  - only relevant when the server runs the multilingual model
  - ignored for English-only model mode
  - must be one of the values from `GET /v1/languages`

- `split_sentences: boolean`
  - default depends on server env, usually `true`
  - if `true`, the server splits the input into sentence chunks
  - this can improve throughput for shorter segments because batching works better

- `exaggeration: float`
  - default: `0.5`
  - controls expressive exaggeration
  - neutral baseline is around `0.5`

- `temperature: float`
  - default: `0.8`
  - sampling temperature for T3 token generation

- `diffusion_steps: integer`
  - default: `4`
  - number of diffusion steps used by S3Gen
  - lower = faster, higher = slower and usually higher quality

- `min_p: float`
  - default: `0.05`
  - vLLM sampling parameter

- `top_p: float`
  - default: `1.0`
  - nucleus sampling parameter

- `repetition_penalty: float`
  - default: `1.2`
  - discourages repetitive token generation

- `seed: integer`
  - default: `0`
  - `0` means effectively random behavior
  - non-zero sets a deterministic seed for the request

## When to Use Form vs Multipart

### Use `application/x-www-form-urlencoded`

Use this when you are not sending `audio_prompt`.

### Use `multipart/form-data`

Use this when you want to upload `audio_prompt`.

That is the recommended path for voice cloning.

## Example Request Without Voice Cloning

Example payload:

```text
text=Hello world&language_id=en&split_sentences=true&exaggeration=0.5&temperature=0.8&diffusion_steps=4&min_p=0.05&top_p=1.0&repetition_penalty=1.2&seed=0
```

## Example Request With Voice Cloning

Example multipart fields:

- `text`
- `audio_prompt`
- `language_id`
- `split_sentences`
- `exaggeration`
- `temperature`
- `diffusion_steps`
- `min_p`
- `top_p`
- `repetition_penalty`
- `seed`

Important:
- the uploaded file must be valid audio
- the file bytes must match the real file format
- do not hand-build malformed multipart bodies
- use `FormData` on the React Native side

## Recommended React Native Flow

### Flow A: plain TTS without voice cloning

1. Check `GET /healthz`
2. Optionally fetch `GET /v1/languages`
3. POST text parameters to `/v1/tts`
4. Receive WAV bytes
5. Save WAV to app storage
6. Play or upload/share/process the file

### Flow B: TTS with voice cloning

1. User picks or records reference audio
2. App builds `FormData`
3. POST to `/v1/tts` with `audio_prompt`
4. Receive WAV bytes
5. Save WAV to app storage
6. Play or use the file elsewhere

## React Native Client Design Notes

Because `/v1/tts` returns binary WAV audio, your app should:

- treat the response as binary, not JSON
- write the bytes to a file
- use the saved file URI for playback or later processing

Depending on your React Native stack, you will usually use one of:

- `react-native-fs`
- `expo-file-system`
- `rn-fetch-blob` or an equivalent binary/network helper
- a player library like `expo-av`, `react-native-track-player`, or `react-native-sound`

## Example: Fetch Languages

```ts
const baseUrl = 'http://192.168.1.23:8000';

export async function fetchLanguages() {
  const res = await fetch(`${baseUrl}/v1/languages`);
  if (!res.ok) {
    throw new Error(`Failed to fetch languages: ${res.status}`);
  }
  return await res.json();
}
```

## Example: Health Check

```ts
const baseUrl = 'http://192.168.1.23:8000';

export async function checkTtsServer() {
  const res = await fetch(`${baseUrl}/healthz`);
  if (!res.ok) {
    return false;
  }
  const data = await res.json();
  return data.status === 'ok';
}
```

## Example: Build a FormData Request

This is the shape to use for voice cloning.

```ts
const form = new FormData();
form.append('text', 'Hello world');
form.append('language_id', 'en');
form.append('split_sentences', 'true');
form.append('exaggeration', '0.5');
form.append('temperature', '0.8');
form.append('diffusion_steps', '4');
form.append('min_p', '0.05');
form.append('top_p', '1.0');
form.append('repetition_penalty', '1.2');
form.append('seed', '0');

form.append('audio_prompt', {
  uri: referenceAudioUri,
  name: 'reference.wav',
  type: 'audio/wav',
} as any);
```

Important:
- React Native file upload objects vary slightly by library/environment
- the `uri`, `name`, and `type` fields are the important parts

## Example: Posting TTS and Saving WAV

The exact implementation depends on your file library, but the overall pattern is:

1. call `/v1/tts`
2. collect binary response
3. save to a local file
4. use the saved URI

Pseudo-example:

```ts
async function generateTts() {
  const form = new FormData();
  form.append('text', 'Hello from React Native');
  form.append('language_id', 'en');
  form.append('split_sentences', 'true');
  form.append('exaggeration', '0.5');
  form.append('temperature', '0.8');
  form.append('diffusion_steps', '4');
  form.append('min_p', '0.05');
  form.append('top_p', '1.0');
  form.append('repetition_penalty', '1.2');
  form.append('seed', '0');

  const response = await fetch('http://192.168.1.23:8000/v1/tts', {
    method: 'POST',
    body: form,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`TTS failed: ${response.status} ${text}`);
  }

  const generationSeconds = response.headers.get('X-Generation-Seconds');
  const audioSeconds = response.headers.get('X-Audio-Seconds');

  const arrayBuffer = await response.arrayBuffer();

  // Convert ArrayBuffer to bytes and write to a local file using your
  // preferred file library.
  return {
    bytes: arrayBuffer,
    generationSeconds,
    audioSeconds,
  };
}
```

## Saving the WAV for Playback or Reuse

Once you receive the WAV bytes, your app can:

- save them to temporary storage for immediate playback
- save them to persistent storage for history/caching
- upload them somewhere else
- attach them to another workflow
- share them to another app

Typical storage uses:

- temporary cache file:
  - best for one-time playback
- app document file:
  - best for history or offline reuse

Recommended filename pattern:

- `tts_<timestamp>.wav`

Example:

- `tts_1712345678901.wav`

## Playback Strategy

Best practice:

1. save the bytes to a local `.wav` file
2. hand the resulting file URI to your audio player

That is usually more reliable than trying to directly play a raw binary response in memory.

## Suggested Frontend State Model

For a clean UX, your React Native app can model the TTS interaction with these states:

- `idle`
- `checkingServer`
- `ready`
- `uploading`
- `generating`
- `savingAudio`
- `readyToPlay`
- `error`

Useful metadata to keep in state:

- `languageId`
- `text`
- `referenceAudioUri`
- `outputAudioUri`
- `generationSeconds`
- `audioSeconds`
- `realtimeFactor`
- `chunks`

## Error Handling Guidance

### Common failure classes

- server offline
- wrong LAN URL
- mobile device cannot reach port `8000`
- invalid `language_id`
- empty `text`
- malformed multipart upload
- invalid reference audio file
- server returns `500`

### What the app should do

If `response.ok` is false:
- read the response body as text
- show a useful message
- keep the original request inputs so the user can retry

Suggested frontend messages:

- "TTS server is offline."
- "Could not upload reference audio."
- "The server rejected the selected language."
- "Audio generation failed. Please try again."

## Recommended Request Defaults for Mobile

A good default mobile request:

- `split_sentences=true`
- `exaggeration=0.5`
- `temperature=0.8`
- `diffusion_steps=4`
- `min_p=0.05`
- `top_p=1.0`
- `repetition_penalty=1.2`
- `seed=0`

These match the current server defaults and are a good starting point.

## Performance Notes for Mobile Apps

- WAV files are large compared to MP3 or M4A.
- Long input text can produce large responses.
- Uploading reference audio plus downloading WAV will increase total latency.
- Sentence splitting may improve throughput depending on input length.

If your app sends a lot of requests:
- debounce rapid user taps
- show progress UI
- save recent outputs locally if replay is common

## Security and Deployment Notes

Current server behavior is development-friendly, not mobile-production-hardened.

If you later expose this beyond your local network, consider:
- HTTPS
- auth tokens
- request limits
- file size limits
- stronger error handling and request validation

## Example Capability Matrix

Your React Native app can currently leverage the server for:

- plain text-to-speech
- multilingual text-to-speech if multilingual model mode is active
- voice cloning via uploaded reference audio
- deterministic generation via `seed`
- quality/speed tuning via `diffusion_steps`
- generation telemetry via response headers

## Integration Checklist

Use this checklist in your app implementation:

- confirm the server is reachable with `GET /healthz`
- fetch supported languages from `GET /v1/languages`
- use the machine LAN IP, not `localhost`, on physical devices
- build `FormData` for voice cloning requests
- treat `/v1/tts` as binary WAV output
- save the WAV to a local file
- pass the file URI to your player or downstream workflow
- handle non-200 responses gracefully

## Bottom Line

The current server is already suitable for React Native integration.

Your app should think of `/v1/tts` as:

- input: text plus optional reference audio and generation controls
- output: a binary WAV file plus useful metadata in headers

That makes the server a good fit for:
- immediate TTS playback
- voice cloning workflows
- saved voice notes
- audio export
- attaching generated speech to other app features
