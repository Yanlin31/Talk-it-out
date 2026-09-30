#!/usr/bin/env node
// Talk to your agent from a browser tab.
//
//   npm start
//
// The API key stays in this process; the page only gets 60-second tokens.

import http from 'node:http'
import { aai, loadEnv, publishAgent, readAgent, required, storedAgentId } from '../../lib.mjs'
import { buildContextPrompt, createPlan, createTask, getState, listSessionIds, recordTurn, resetPersisted, saveIdea } from './organizer.mjs'

loadEnv()
required('ASSEMBLYAI_API_KEY', 'get one at https://www.assemblyai.com/dashboard/api-keys')

// A published id means the agent is managed elsewhere, so use it as it is.
const AGENT = await (async () => {
  const name = process.env.AGENT || 'minimal'
  const known = storedAgentId(name)
  if (known) {
    try {
      const agent = await aai(`/agents/${known}`)
      return { id: known, name: agent.name || 'Your agent' }
    } catch (error) {
      console.error(`Could not load agent ${known}: ${error.message}`)
      process.exit(1)
    }
  }
  const agent = readAgent(name)
  try {
    const { id, created } = await publishAgent(agent, { name, reuseByName: true })
    console.log(`${created ? 'Created' : 'Updated'} "${agent.name}" from agents/${name}.jsonc`)
    return { id, name: agent.name }
  } catch (error) {
    console.error(`Could not publish agents/${name}.jsonc: ${error.message}`)
    process.exit(1)
  }
})()

console.log(`Agent: ${AGENT.id}`)

// The prompt the agent actually runs with today. Module 3 re-sends it with a
// summary of the organizer appended, because a mid-call update replaces the
// whole prompt rather than adding to it.
const BASE_PROMPT = await (async () => {
  try {
    const stored = await aai(`/agents/${AGENT.id}`)
    if (stored.system_prompt) return stored.system_prompt
  } catch {}
  try {
    return readAgent(process.env.AGENT || 'minimal').system_prompt || ''
  } catch {
    return ''
  }
})()

// --- client ----------------------------------------------------------------
// Stringified and served as /app.js.
function clientApp() {
const $ = (id) => document.getElementById(id)
// The rate the API speaks. Both worklets resample, since a browser may
// ignore the rate an AudioContext asks for.
const WIRE_RATE = 24_000
const AGENT = window.AGENT

// Scratch buffers are reused: allocating on the audio thread causes glitches.
const CAPTURE_WORKLET = `
  class CaptureProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this._ratio = sampleRate / ${WIRE_RATE};
      this._pos = 0;
      this._prev = 0;
      this._src = null;
      this._out = null;
    }
    _toPcm(samples, len) {
      const pcm = new Int16Array(len);
      for (let i = 0; i < len; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      return pcm;
    }
    process(inputs) {
      const ch = inputs[0]?.[0];
      if (!ch) return true;
      if (this._ratio === 1) {
        const pcm = this._toPcm(ch, ch.length);
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        return true;
      }
      const n = ch.length;
      if (!this._src || this._src.length < n + 1) {
        this._src = new Float32Array(n + 1);
        this._out = new Float32Array(Math.ceil((n + 1) / this._ratio) + 2);
      }
      const src = this._src;
      const out = this._out;
      src[0] = this._prev;
      src.set(ch, 1);
      let outLen = 0;
      let pos = this._pos;
      while (pos < n) {
        const i = Math.floor(pos);
        const frac = pos - i;
        out[outLen++] = src[i] + (src[i + 1] - src[i]) * frac;
        pos += this._ratio;
      }
      this._pos = pos - n;
      this._prev = ch[n - 1];
      if (outLen) {
        const pcm = this._toPcm(out, outLen);
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
      }
      return true;
    }
  }
  registerProcessor('capture', CaptureProcessor);
`

// A ring buffer rather than one AudioBufferSource per chunk, which drifts and
// clicks under jitter. Posting 'stop' empties it for barge-in.
const PLAYBACK_WORKLET = `
  class PlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this._ring = new Float32Array(sampleRate * 30);
      this._writePos = 0;
      this._readPos = 0;
      this._available = 0;
      this._step = ${WIRE_RATE} / sampleRate;
      this._rsPos = 0;
      this._rsPrev = 0;
      // After a gap the speaker sits at zero, so interpolating from the
      // pre-gap _rsPrev would click. Reset it instead.
      this._drained = false;
      this.port.onmessage = (e) => {
        if (e.data === 'stop') {
          this._writePos = this._readPos = this._available = 0;
          this._rsPos = this._rsPrev = 0;
          return;
        }
        const int16 = new Int16Array(e.data);
        // int16[-1] would make _rsPrev NaN, silencing the ring for good.
        if (!int16.length) return;
        if (this._drained) {
          this._rsPrev = 0;
          this._rsPos = 0;
          this._drained = false;
        }
        if (this._step === 1) {
          for (let i = 0; i < int16.length; i++) this._push(int16[i] / 32768);
          return;
        }
        const n = int16.length;
        let pos = this._rsPos;
        while (pos < n) {
          const i = Math.floor(pos);
          const frac = pos - i;
          const a = i === 0 ? this._rsPrev : int16[i - 1] / 32768;
          const b = int16[i] / 32768;
          this._push(a + (b - a) * frac);
          pos += this._step;
        }
        this._rsPos = pos - n;
        this._rsPrev = int16[n - 1] / 32768;
      };
    }
    _push(v) {
      if (this._available < this._ring.length) {
        this._ring[this._writePos] = v;
        this._writePos = (this._writePos + 1) % this._ring.length;
        this._available++;
      }
    }
    process(inputs, outputs) {
      const output = outputs[0];
      const out = output[0];
      const cap = this._ring.length;
      for (let i = 0; i < out.length; i++) {
        if (this._available > 0) {
          out[i] = this._ring[this._readPos];
          this._readPos = (this._readPos + 1) % cap;
          this._available--;
        } else {
          out[i] = 0;
          this._drained = true;
        }
      }
      // Mono source, stereo sink.
      for (let ch = 1; ch < output.length; ch++) output[ch].set(out);
      return true;
    }
  }
  registerProcessor('playback', PlaybackProcessor);
`

const blobUrl = (code) =>
  URL.createObjectURL(new Blob([code], { type: 'application/javascript' }))

let ws, captureCtx, playbackCtx, playback, mic, callStart, timer
let sessionId = null
let lastPrompt = null

// Module 5: as soon as the call connects, push whatever is already known
// from past sessions — before the first turn, not after — so continuity is
// audible from the greeting onward.
async function pushInitialContext() {
  const forSession = sessionId
  try {
    const [ctxRes, stateRes] = await Promise.all([
      fetch('/api/context?session_id=' + encodeURIComponent(forSession)),
      fetch('/api/state?session_id=' + encodeURIComponent(forSession)),
    ])
    if (forSession !== sessionId) return
    const { system_prompt } = await ctxRes.json()
    renderOrganizer(await stateRes.json())
    if (!system_prompt || !ws || ws.readyState !== 1) return
    lastPrompt = system_prompt
    ws.send(JSON.stringify({ type: 'session.update', session: { system_prompt } }))
    logEvent('up', 'session.update', 'organizer context (from memory)')
  } catch (error) {
    console.warn('[organizer] could not load prior context:', error.message)
  }
}

let pendingToolResults = []
const KNOWN_TOOLS = new Set(['save_idea', 'create_task', 'create_plan'])

async function handleToolCall(msg) {
  if (!KNOWN_TOOLS.has(msg.name) || !sessionId) return
  const forSession = sessionId
  let args = msg.arguments
  if (typeof args === 'string') {
    try { args = JSON.parse(args) } catch { args = {} }
  }
  try {
    const res = await fetch('/api/tool', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ session_id: forSession, name: msg.name, arguments: args || {} }),
    })
    const { state, system_prompt, message } = await res.json()
    if (forSession !== sessionId) return // a new call started while this was in flight
    if (state) renderOrganizer(state)
    if (system_prompt && system_prompt !== lastPrompt && ws && ws.readyState === 1) {
      lastPrompt = system_prompt
      ws.send(JSON.stringify({ type: 'session.update', session: { system_prompt } }))
      logEvent('up', 'session.update', 'organizer context')
    }
    pendingToolResults.push({ call_id: msg.call_id, result: message || 'Done.' })
  } catch (error) {
    console.warn('[organizer] tool call failed:', error.message)
    pendingToolResults.push({ call_id: msg.call_id, result: "Sorry, I couldn't do that just now." })
  }
}

// Fire-and-forget: organizing the conversation should never slow down or
// break the conversation itself.
async function relayTurn(role, text) {
  if (!sessionId || !text) return
  const forSession = sessionId
  try {
    const res = await fetch('/api/turn', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, role, text }),
    })
    if (!res.ok) return
    const { state, system_prompt } = await res.json()
    if (forSession !== sessionId) return // a new call started while this was in flight
    renderOrganizer(state)
    // Module 3: keep the agent aware of what is in the panel. Only sent when
    // the prompt actually changed.
    if (system_prompt && system_prompt !== lastPrompt && ws && ws.readyState === 1) {
      lastPrompt = system_prompt
      ws.send(JSON.stringify({ type: 'session.update', session: { system_prompt } }))
      logEvent('up', 'session.update', 'organizer context')
    }
    // Also logged as text so it's copy-pasteable from devtools.
    console.log(`[organizer] session=${sessionId}\n` + JSON.stringify(state, null, 2))
  } catch (error) {
    console.warn('[organizer] relay failed:', error.message)
  }
}

// --- microphones ---
// Labels stay empty until mic permission is granted, so this runs again after
// getUserMedia.
async function listMics() {
  if (!navigator.mediaDevices?.enumerateDevices) return
  const devices = await navigator.mediaDevices.enumerateDevices()
  const inputs = devices
    .filter((device) => device.kind === 'audioinput')
    // Chrome's synthetic entries alias a real device and duplicate it.
    .filter((device) => device.deviceId !== 'default' && device.deviceId !== 'communications')
  const select = $('mic')
  const chosen = select.value
  select.replaceChildren()
  const auto = document.createElement('option')
  auto.value = ''
  auto.textContent = 'Default microphone'
  select.append(auto)
  inputs.forEach((device, i) => {
    const option = document.createElement('option')
    option.value = device.deviceId
    option.textContent = device.label || `Microphone ${i + 1}`
    select.append(option)
  })
  if (chosen && inputs.some((device) => device.deviceId === chosen)) select.value = chosen
}
listMics()
navigator.mediaDevices?.addEventListener?.('devicechange', listMics)

$('btn').onclick = () => (ws?.readyState <= 1 ? stop() : start())
$('log-toggle').onclick = () => {
  const hidden = document.body.classList.toggle('no-side')
  $('log-toggle').textContent = hidden ? 'Show' : 'Hide'
}

// --- thought organizer ---
// Renders the state the extraction pipeline builds. New items fade in so the
// AI's understanding visibly grows as you talk. Text is set via textContent
// only — model output is never treated as HTML.
const seen = new Set()
const SECTIONS = [
  ['mainGoal', 'Main goal'],
  ['tasks', 'Tasks'],
  ['ideas', 'Ideas'],
  ['questions', 'Questions'],
  ['conflicts', 'Conflicts'],
  ['concerns', 'Concerns'],
  ['connections', 'Connections'],
  ['goals', 'Goals'],
  ['suggestions', 'Suggestions'],
  ['nextSteps', 'Next steps'],
]

function thoughtItem(key, item) {
  if (typeof item === 'string') return { text: item }
  if (key === 'tasks') {
    const meta = [item.deadline, item.priority && item.priority + ' priority'].filter(Boolean).join(' · ')
    return { text: item.title, meta, done: item.completed, id: item.id }
  }
  return { text: item.title, meta: item.description, id: item.id }
}

function renderOrganizer(state) {
  const root = $('thoughts-body')
  root.replaceChildren()
  let any = false
  for (const [key, label] of SECTIONS) {
    const items = key === 'mainGoal' ? (state.mainGoal ? [state.mainGoal] : []) : state[key] || []
    if (!items.length) continue
    any = true
    const section = document.createElement('section')
    section.className = 'th-sec'
    const heading = document.createElement('h3')
    heading.textContent = label
    section.append(heading)
    for (const raw of items) {
      const item = thoughtItem(key, raw)
      const id = key + ':' + (item.id || item.text)
      const row = document.createElement('div')
      row.className = 'th-item ' + key + (item.done ? ' done' : '')
      if (!seen.has(id)) {
        seen.add(id)
        row.classList.add('fresh')
      }
      const text = document.createElement('div')
      text.className = 'th-text'
      text.textContent = item.text
      row.append(text)
      if (item.meta) {
        const meta = document.createElement('div')
        meta.className = 'th-meta'
        meta.textContent = item.meta
        row.append(meta)
      }
      section.append(row)
    }
    root.append(section)
  }
  if (!any) {
    const empty = document.createElement('div')
    empty.className = 'empty'
    empty.textContent = "Just talk. I'll organize your thoughts here as you go."
    root.append(empty)
  }
}

// --- side pane tabs ---
let agentLoaded = false

function showTab(name) {
  for (const tab of ['thoughts', 'events', 'agent']) {
    $('tab-' + tab).classList.toggle('on', tab === name)
    $(tab + '-body').hidden = tab !== name
  }
  if (name === 'agent' && !agentLoaded) {
    agentLoaded = true
    fetch('/agent')
      .then((res) => res.json())
      .then((agent) => {
        $('agent-body').replaceChildren()
        const pre = document.createElement('pre')
        pre.textContent = JSON.stringify(agent, null, 2)
        $('agent-body').append(pre)
      })
      .catch(() => {
        agentLoaded = false
        $('agent-body').textContent = 'Could not load the agent.'
      })
  }
}
$('mem-reset').onclick = async () => {
  if (!confirm('Forget everything Talk It Out has organized so far? This clears saved memory on disk too.')) return
  await fetch('/api/reset', { method: 'POST' })
  seen.clear()
  renderOrganizer({})
}
$('tab-thoughts').onclick = () => showTab('thoughts')
$('tab-events').onclick = () => showTab('events')
$('tab-agent').onclick = () => showTab('agent')

async function addWorklet(ctx, code, name) {
  const url = blobUrl(code)
  try {
    await ctx.audioWorklet.addModule(url)
  } finally {
    URL.revokeObjectURL(url)
  }
  return new AudioWorkletNode(ctx, name)
}

async function start() {
  $('btn').disabled = true
  $('mic').disabled = true
  setStatus('connecting')

  try {
    // The API key never reaches the page; this token expires in 60 seconds.
    const res = await fetch('/token')
    if (!res.ok) {
      setStatus('error', 'could not mint a token, check the API key')
      reset()
      return
    }
    const { token } = await res.json()

    // Two contexts, created in the click handler so Safari starts them.
    captureCtx = new AudioContext({ sampleRate: WIRE_RATE })
    playbackCtx = new AudioContext({ sampleRate: WIRE_RATE })
    await Promise.all([captureCtx.resume(), playbackCtx.resume()])

    playback = await addWorklet(playbackCtx, PLAYBACK_WORKLET, 'playback')
    playback.connect(playbackCtx.destination)

    const deviceId = $('mic').value
    mic = await navigator.mediaDevices.getUserMedia({
      audio: {
        // A preference, not `exact`: an unplugged device falls back.
        ...(deviceId ? { deviceId } : {}),
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: false,
        autoGainControl: false,
      },
    })
    listMics()
    const capture = await addWorklet(captureCtx, CAPTURE_WORKLET, 'capture')
    captureCtx.createMediaStreamSource(mic).connect(capture)

    const url = new URL('wss://agents.assemblyai.com/v1/ws')
    url.searchParams.set('token', token)
    ws = new WebSocket(url)
    let ready = false

    // The API takes base64 inside JSON, not binary frames.
    capture.port.onmessage = ({ data }) => {
      if (!ready || ws.readyState !== 1) return
      const bytes = new Uint8Array(data)
      let binary = ''
      for (let i = 0; i < bytes.length; i += 0x8000) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000))
      }
      ws.send(JSON.stringify({ type: 'input.audio', audio: btoa(binary) }))
      logEvent('up', 'input.audio')
    }

    // Everything about the agent lives server-side; the session just names it.
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'session.update', session: { agent_id: AGENT.id } }))
      logEvent('up', 'session.update', AGENT.id)
    }

    ws.onmessage = ({ data }) => {
      const msg = JSON.parse(data)
      switch (msg.type) {
        case 'session.ready':
          ready = true
          sessionId = msg.session_id
          seen.clear()
          lastPrompt = null
          renderOrganizer({})
          pushInitialContext()
          callStart = Date.now()
          timer = setInterval(tick, 1000)
          tick()
          setStatus('listening')
          $('btn').disabled = false
          $('btn').textContent = 'End call'
          $('btn').classList.add('live')
          logEvent('down', msg.type, msg.session_id)
          break

        case 'input.speech.started':
          // Barge-in: empty the ring buffer so the agent stops mid-word.
          playback?.port.postMessage('stop')
          setStatus('listening')
          logEvent('down', msg.type)
          break

        case 'reply.started':
          setStatus('speaking')
          logEvent('down', msg.type)
          break

        case 'reply.audio': {
          const raw = atob(msg.data)
          const bytes = new Uint8Array(raw.length)
          for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i)
          playback?.port.postMessage(bytes.buffer, [bytes.buffer])
          logEvent('down', msg.type)
          break
        }

        case 'reply.done':
          setStatus('listening')
          if (msg.status === 'interrupted') {
            playback?.port.postMessage('stop')
            pendingToolResults = [] // discard — the reply they were attached to never finished
          } else {
            for (const { call_id, result } of pendingToolResults) {
              ws.send(JSON.stringify({ type: 'tool.result', call_id, result }))
              logEvent('up', 'tool.result', result)
            }
            pendingToolResults = []
          }
          logEvent('down', msg.type, msg.status)
          break

        // text is the full transcript so far, so it replaces.
        case 'transcript.user.delta':
          partial('you', msg.text)
          logEvent('down', msg.type, msg.text)
          break

        // delta is the next word only, so it appends.
        case 'transcript.agent.delta':
          logEvent('down', msg.type, msg.delta)
          if (msg.reply_id && msg.reply_id === printedReply) break
          if (msg.reply_id !== liveReply) {
            liveReply = msg.reply_id
            dropPartial('agent')
          }
          partial('agent', appendDelta(partialText.agent || '', msg.delta))
          break

        case 'transcript.user':
          addLine('you', msg.text)
          logEvent('down', msg.type, msg.text)
          relayTurn('user', msg.text)
          break

        case 'transcript.agent':
          printedReply = msg.reply_id ?? printedReply
          addLine('agent', msg.text)
          logEvent('down', msg.type, msg.text)
          relayTurn('agent', msg.text)
          break

        case 'tool.call': {
          // http tools (if any are ever added) run on AssemblyAI's side and
          // expect no result here — this only executes our own function
          // tools (save_idea, create_task, create_plan).
          const args = JSON.stringify(msg.arguments ?? {})
          addLine('tool', `${msg.name}(${args})`)
          logEvent('down', msg.type, `${msg.name} ${args}`)
          handleToolCall(msg)
          break
        }

        case 'session.ended':
          logEvent('down', msg.type)
          ws.close()
          break

        case 'session.error':
          setStatus('error', msg.message)
          logEvent('down', msg.type, `${msg.code}: ${msg.message}`)
          break

        default:
          logEvent('down', msg.type)
      }
    }

    ws.onclose = () => { setStatus('idle'); reset() }
    ws.onerror = () => { setStatus('error', 'connection failed'); reset() }
  } catch (error) {
    setStatus('error', error.message)
    reset()
  }
}

function stop() {
  // Close cleanly so the session record ends, falling back to the socket.
  if (ws?.readyState === 1) {
    ws.send(JSON.stringify({ type: 'session.end' }))
    logEvent('up', 'session.end')
    const socket = ws
    setTimeout(() => { if (socket.readyState === 1) socket.close() }, 3000)
  } else {
    ws?.close()
  }
  playback?.port.postMessage('stop')
  mic?.getTracks().forEach((track) => track.stop())
  captureCtx?.close()
  playbackCtx?.close()
  captureCtx = playbackCtx = playback = mic = null
  reset()
  setStatus('idle')
}

function reset() {
  clearInterval(timer)
  clearPartials()
  open.forEach((run) => paint(run, true))
  open.clear()
  $('btn').disabled = false
  $('mic').disabled = false
  $('btn').textContent = 'Start call'
  $('btn').classList.remove('live')
}

function setStatus(state, detail) {
  $('status').className = 'status ' + state
  $('status-text').textContent = detail || state
}

// $4.50 an hour, the list price at assemblyai.com/pricing. Billing is per
// session minute, so the running figure is an estimate, not an invoice.
const COST_PER_SECOND = 4.5 / 3600

function tick() {
  const seconds = Math.floor((Date.now() - callStart) / 1000)
  $('elapsed').textContent =
    Math.floor(seconds / 60) + ':' + String(seconds % 60).padStart(2, '0')
  $('cost').textContent = '$' + (seconds * COST_PER_SECOND).toFixed(3)
}

// --- transcript ---
const partialText = {}
const partialEl = {}
// The full reply arrives once its audio has been sent, which beats the audio
// playing out, so deltas keep coming after the line is printed. printedReply
// stops them rebuilding the same sentence underneath it.
let liveReply = null
let printedReply = null

// Deltas arrive with a leading space sometimes and without it other times, so
// add one only when neither side has one and the delta is not punctuation.
const ATTACHES_LEFT = /^[.,!?;:%°)\]}…'"’”]/
const NO_SPACE_AFTER = /[([{$\-\/'"‘“]$/

function appendDelta(text, delta) {
  if (!delta) return text
  if (!text) return delta
  if (/^\s/.test(delta) || /\s$/.test(text)) return text + delta
  if (ATTACHES_LEFT.test(delta) || NO_SPACE_AFTER.test(text)) return text + delta
  return text + ' ' + delta
}

function dropPartial(who) {
  partialEl[who]?.remove()
  delete partialEl[who]
  delete partialText[who]
}

function transcriptLine(who, text, cls) {
  const line = document.createElement('div')
  line.className = 'line ' + who + (cls ? ' ' + cls : '')
  const label = document.createElement('span')
  label.className = 'who'
  label.textContent = who === 'agent' ? AGENT.name : who
  const body = document.createElement('span')
  body.className = 'said'
  body.textContent = text
  line.append(label, body)
  return line
}

function clearEmpty(el) {
  const empty = el.querySelector('.empty')
  if (empty) empty.remove()
}

function scroll(el) {
  el.scrollTop = el.scrollHeight
}

function partial(who, text) {
  clearEmpty($('transcript'))
  partialText[who] = text
  if (partialEl[who]) {
    partialEl[who].querySelector('.said').textContent = text
  } else {
    partialEl[who] = transcriptLine(who, text, 'partial')
    $('transcript').append(partialEl[who])
  }
  scroll($('transcript'))
}

function addLine(who, text) {
  clearEmpty($('transcript'))
  dropPartial(who)
  $('transcript').append(transcriptLine(who, text))
  scroll($('transcript'))
}

function clearPartials() {
  for (const who of Object.keys(partialEl)) dropPartial(who)
  liveReply = printedReply = null
}

// --- event log ---
// Audio frames arrive ~190 times a second each way, so these types hold a row
// open and count into it. Both streams run at once, hence a row per key.
const COALESCE = new Set([
  'input.audio',
  'reply.audio',
  'transcript.user.delta',
  'transcript.agent.delta',
])
const open = new Map()

function eventRow(direction, type, detail) {
  const row = document.createElement('div')
  row.className = 'event ' + direction
  const at = document.createElement('span')
  at.className = 'at'
  at.textContent = (callStart ? (Date.now() - callStart) / 1000 : 0).toFixed(1) + 's'
  const arrow = document.createElement('span')
  arrow.className = 'dir'
  arrow.textContent = direction === 'up' ? '↑' : '↓'
  const name = document.createElement('span')
  name.className = 'type'
  name.textContent = type
  const count = document.createElement('span')
  count.className = 'count'
  const info = document.createElement('span')
  info.className = 'detail'
  if (detail) info.textContent = detail
  row.append(at, arrow, name, count, info)
  return row
}

// Ten repaints a second, plus one when the run closes.
function paint(live, final) {
  const now = performance.now()
  if (!final && now - live.painted < 100) return
  live.painted = now
  live.row.querySelector('.count').textContent = live.count > 1 ? '×' + live.count : ''
  if (live.detail) live.row.querySelector('.detail').textContent = live.detail
}

function logEvent(direction, type, detail) {
  const log = $('events-body')
  clearEmpty(log)
  const key = direction + ' ' + type
  const live = open.get(key)
  if (live) {
    live.count += 1
    if (detail) live.detail = detail
    paint(live)
    return
  }
  // A real event closes the open runs, so the next burst starts a new row.
  if (!COALESCE.has(type)) {
    open.forEach((run) => paint(run, true))
    open.clear()
  }
  // Only follow the tail if the reader is there.
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40
  const row = eventRow(direction, type, detail)
  log.append(row)
  while (log.children.length > 400) log.firstChild.remove()
  if (COALESCE.has(type)) open.set(key, { row, count: 1, detail, painted: 0 })
  if (atBottom) scroll(log)
}
}

// --- page ------------------------------------------------------------------
const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${AGENT.name}</title>
<style>
  /* Tokens taken from assemblyai.com. The three typefaces are licensed and
     not bundled here, so each falls back the same way the site's own stack
     does: Georgia for display, system-ui for body, JetBrains Mono for mono. */
  :root {
    --page-bg: #fdfcf8;
    --surface: #fff;
    --surface-alt: #f5f3eb;
    --border: #dad7cb;
    --border-strong: #c7c3b2;
    --text: #4a4945;
    --text-dark: #1d1b16;
    --text-muted: #777673;
    --text-faint: #a5a4a2;
    --cobolt-500: #3923c7;
    --cobolt-300: #887bdd;
    --cobolt-100: #d7d3f4;
    --green-500: #01762f;
    --error: #f04438;
    --radius-sm: 4px;
    --radius-lg: 12px;
    --font-display: "Oceanic Text", Georgia, serif;
    --font-body: "UN 11ST", system-ui, -apple-system, sans-serif;
    --font-mono: "Modern Gothic Mono", "JetBrains Mono", ui-monospace, monospace;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    font-family: var(--font-body); font-size: 16px; line-height: 1.3;
    color: var(--text); background: var(--page-bg); display: flex;
    flex-direction: column; align-items: center; padding: 24px 20px 20px;
  }
  main { width: 100%; max-width: 1088px; flex: 1; display: flex;
         flex-direction: column; min-height: 0; gap: 16px; }

  /* .eyebrow on the site: mono, 12px, uppercase, 1.2px tracking. */
  .eyebrow { font-family: var(--font-mono); font-size: 12px; letter-spacing: 1.2px;
             text-transform: uppercase; font-feature-settings: "ss09" 1; }

  header { display: flex; align-items: center; gap: 16px;
           padding-bottom: 16px; border-bottom: 1px solid var(--border); }
  h1 { font-family: var(--font-display); font-size: 24px; font-weight: 400;
       letter-spacing: -1.2px; line-height: 1; color: var(--text-dark);
       margin-right: auto; }
  .status { display: flex; align-items: center; gap: 8px; color: var(--text-muted); }
  .status::before { content: ""; width: 7px; height: 7px; border-radius: 50%;
                    background: currentColor; flex-shrink: 0; }
  .status.listening { color: var(--green-500); }
  .status.speaking { color: var(--cobolt-500); }
  .status.error { color: var(--error); text-transform: none; letter-spacing: 0;
                  font-family: var(--font-body); font-size: 14px; }
  .status.listening::before, .status.speaking::before {
    animation: pulse 1.6s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1 } 50% { opacity: .25 } }
  .meter { display: flex; gap: 10px; font-family: var(--font-mono); font-size: 12px;
           color: var(--text-faint); }
  #elapsed { min-width: 34px; text-align: right; }
  #cost { min-width: 48px; text-align: right; }

  .panes { flex: 1; min-height: 0; display: grid; gap: 16px;
           grid-template-columns: 1fr 420px; }
  body.no-side .panes { grid-template-columns: 1fr; }
  body.no-side #side { display: none; }
  [hidden] { display: none !important; }
  @media (max-width: 880px) {
    .panes { grid-template-columns: 1fr; grid-template-rows: 1fr 176px; }
    body.no-side .panes { grid-template-rows: 1fr; }
  }

  .pane { display: flex; flex-direction: column; min-height: 0;
          background: var(--surface); border: 1px solid var(--border);
          border-radius: var(--radius-lg); overflow: hidden; }
  .pane-head { display: flex; align-items: center; justify-content: space-between;
               gap: 16px; padding: 10px 16px; background: var(--surface-alt);
               border-bottom: 1px solid var(--border); color: var(--text-muted); }
  .pane-body { flex: 1; overflow-y: auto; padding: 16px; }
  .empty { color: var(--text-faint); font-size: 14px; line-height: 1.4; }

  #transcript { display: flex; flex-direction: column; gap: 12px; }
  .line { display: flex; gap: 12px; font-size: 16px; line-height: 1.4; }
  .who { color: var(--text-faint); padding-top: 3px; flex-shrink: 0; width: 88px;
         overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .line.agent .said { color: var(--text-dark); }
  .line.partial .said { color: var(--text-muted); }
  .line.tool { font-family: var(--font-mono); font-size: 13px;
               color: var(--cobolt-500); }
  .line.tool .said { word-break: break-all; }

  #events-body { font-family: var(--font-mono); font-size: 12px; line-height: 1.8; }
  .event { display: flex; gap: 8px; align-items: baseline; white-space: nowrap; }
  .event .at { color: var(--text-faint); min-width: 44px; text-align: right;
               flex-shrink: 0; }
  .event .dir, .event .count { color: var(--text-faint); flex-shrink: 0; }
  .event .count:empty, .event .detail:empty { display: none; }
  .event .type { flex-shrink: 0; color: var(--text-dark); }
  .event.up .type { color: var(--text-muted); }
  .event .detail { color: var(--text-faint); overflow: hidden; white-space: nowrap;
                   text-overflow: ellipsis; }

  .pane-foot { display: flex; gap: 8px; align-items: center; padding: 12px 16px;
               background: var(--surface-alt); border-top: 1px solid var(--border); }
  /* .cta-primary on the site: cobolt fill, mono uppercase 14px, 1.4px
     tracking, 40px tall, 4px radius, lightening on hover. */
  button { height: 40px; padding: 0 24px; margin-left: auto; border: none;
           border-radius: var(--radius-sm); background: var(--cobolt-500);
           color: #fff; font-family: var(--font-mono); font-size: 14px;
           letter-spacing: 1.4px; text-transform: uppercase; white-space: nowrap;
           cursor: pointer; transition: background-color .2s; }
  button:hover:not(:disabled) { background: var(--cobolt-300); }
  button:disabled { opacity: .55; cursor: default; }
  button.live { background: var(--error); }
  button.live:hover { background: #f4695f; }
  select { flex: 0 1 220px; min-width: 0; height: 40px; padding: 0 8px;
           font-family: var(--font-body); font-size: 13px; color: var(--text-muted);
           background: var(--surface); border: 1px solid var(--border);
           border-radius: var(--radius-sm); }
  select:disabled { color: var(--text-faint); }
  /* Text button, sized to sit inside the pane header. */
  .ghost { height: auto; margin-left: 0; padding: 0; background: transparent;
           color: var(--text-faint); font-size: 12px; letter-spacing: 1.2px; }
  .ghost:hover:not(:disabled) { background: transparent; color: var(--cobolt-500); }
  .tabs { display: flex; gap: 16px; }
  .tab.on { color: var(--text-dark); }

  .subtitle { font-size: 13px; color: var(--text-muted); margin-top: -6px; margin-bottom: 4px; }

  /* Thought organizer: a picture of what the AI currently understands, not a
     task manager — quiet cards, items fade in as they are picked up. */
  .th-sec { margin-bottom: 18px; }
  .th-sec h3 { font-family: var(--font-mono); font-size: 11px; font-weight: 400;
               letter-spacing: 1.2px; text-transform: uppercase;
               color: var(--text-faint); margin-bottom: 8px; }
  .th-item { padding: 8px 10px; margin-bottom: 6px; background: var(--surface);
             border: 1px solid var(--border); border-radius: var(--radius-sm);
             font-size: 14px; line-height: 1.35; color: var(--text-dark); }
  .th-item.mainGoal { font-size: 16px; background: #f4f2fd; border-color: var(--cobolt-300); }
  .th-item.conflicts { border-left: 3px solid var(--error); }
  .th-item.suggestions, .th-item.nextSteps { border-left: 3px solid var(--cobolt-300); }
  .th-item.tasks .th-text::before { content: "\\25CB  "; color: var(--text-faint); }
  .th-item.tasks.done .th-text::before { content: "\\25CF  "; }
  .th-item.done .th-text { text-decoration: line-through; color: var(--text-faint); }
  .th-meta { margin-top: 3px; font-size: 12px; color: var(--text-muted); }
  .th-item.fresh { animation: thIn .6s ease-out; }
  @keyframes thIn {
    from { opacity: 0; transform: translateY(6px); background: var(--cobolt-100); }
    to { opacity: 1; transform: none; }
  }

  /* Read-only view of the agent as the API stored it. */
  #agent-body pre { font-family: var(--font-mono); font-size: 12px;
                    line-height: 1.6; color: var(--text); white-space: pre-wrap;
                    word-break: break-word; }
</style>
</head>
<body>
<main>
  <header>
    <h1>Talk It Out</h1>
    <p class="subtitle">Just talk. I'll help organize your thoughts.</p>
    <span class="status idle" id="status"><span id="status-text">idle</span></span>
    <span class="meter"><span id="elapsed">0:00</span><span id="cost">$0.000</span></span>
  </header>

  <div class="panes">
    <section class="pane">
      <div class="pane-head"><span>Transcript</span></div>
      <div class="pane-body" id="transcript">
        <div class="empty">Just talk. I'll help organize your thoughts as we go, and I'll remember them for next time too.</div>
      </div>
      <div class="pane-foot">
        <select id="mic" aria-label="Microphone"><option value="">Default microphone</option></select>
        <button id="btn">🎙️ What's on your mind?</button>
      </div>
    </section>
    <section class="pane" id="side">
      <div class="pane-head">
        <span class="tabs">
          <button class="ghost tab on" id="tab-thoughts">Your thoughts</button>
          <button class="ghost tab" id="tab-events">Events</button>
          <button class="ghost tab" id="tab-agent">Agent</button>
        </span>
<button class="ghost" id="mem-reset" title="Clear everything Talk It Out remembers, on disk and in this call">Forget everything</button>
        <button class="ghost" id="log-toggle">Hide</button>
      </div>
      <div class="pane-body" id="thoughts-body">
        <div class="empty">Just talk. I'll organize your thoughts here as you go.</div>
      </div>
      <div class="pane-body" id="events-body" hidden>
        <div class="empty">Every websocket frame, both directions. Repeats collapse into a count.</div>
      </div>
      <div class="pane-body" id="agent-body" hidden>
        <div class="empty">Loading the published agent.</div>
      </div>
    </section>
  </div>
</main>
<script>window.AGENT = ${JSON.stringify(AGENT).replace(/</g, '\\u003c')}</script>
<script src="/app.js"></script>
</body>
</html>`

// --- server ----------------------------------------------------------------

// Read-only view of the stored agent. The API keeps header values and llm keys
// write-only; these deletes hold even if that changes. The system prompt is in
// here, so a public deployment shows it to anyone who opens the page.
function publicAgent(agent) {
  const copy = structuredClone(agent)
  for (const tool of copy.tools ?? []) {
    for (const header of tool.http?.headers ?? []) header.value = '<hidden>'
  }
  for (const llm of copy.llm ?? []) delete llm.api_key
  return copy
}

const server = http.createServer(async (req, res) => {
  if (req.url === '/agent') {
    try {
      const agent = await aai(`/agents/${AGENT.id}`)
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify(publicAgent(agent)))
    } catch (error) {
      console.error(error.message)
      res.writeHead(502, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ error: 'could not load the agent' }))
    }
    return
  }
  if (req.url === '/token') {
    try {
      const token = await aai('/token?product=voice_agent&expires_in_seconds=60')
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify(token))
    } catch (error) {
      console.error(error.message)
      res.writeHead(502, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ error: 'token request failed' }))
    }
    return
  }
  // Talk It Out: the browser posts each finalized turn here so the
  // extraction call — and the API key it needs — stays server-side.
  if (req.url === '/api/turn' && req.method === 'POST') {
    let body = ''
    req.on('data', (chunk) => (body += chunk))
    req.on('end', async () => {
      try {
        const { session_id, role, text } = JSON.parse(body)
        if (!session_id || !text || (role !== 'user' && role !== 'agent')) {
          res.writeHead(400, { 'content-type': 'application/json' })
          res.end(JSON.stringify({ error: 'session_id, role (user|agent), and text are required' }))
          return
        }
        console.log(`[organizer] relayed ${role}: ${text}`)
        const state = await recordTurn(session_id, role, text)
        res.writeHead(200, { 'content-type': 'application/json' })
        res.end(JSON.stringify({ state, system_prompt: buildContextPrompt(BASE_PROMPT, state) }))
      } catch (error) {
        console.error('[organizer] turn processing failed:', error.message)
        res.writeHead(502, { 'content-type': 'application/json' })
        res.end(JSON.stringify({ error: 'could not process turn' }))
      }
    })
    return
  }

  // Module 5: wipe persisted memory (dev/demo convenience — a fresh start).
  if (req.url === '/api/reset' && req.method === 'POST') {
    resetPersisted()
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ ok: true }))
    return
  }

  // Immediate awareness on connect, before the first turn even completes —
  // this is what makes persisted memory actually felt, not just present.
  if (req.url.startsWith('/api/context')) {
    const sessionId = new URL(req.url, 'http://x').searchParams.get('session_id')
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ system_prompt: sessionId ? buildContextPrompt(BASE_PROMPT, getState(sessionId)) : null }))
    return
  }

  // Module 4: the agent's own tool calls land here (relayed from the
  // browser, since only the server holds the extraction/organizer state).
  if (req.url === '/api/tool' && req.method === 'POST') {
    let body = ''
    req.on('data', (chunk) => (body += chunk))
    req.on('end', async () => {
      try {
        const { session_id, name, arguments: args } = JSON.parse(body)
        if (!session_id || !name) {
          res.writeHead(400, { 'content-type': 'application/json' })
          res.end(JSON.stringify({ error: 'session_id and name are required' }))
          return
        }
        console.log(`[organizer] tool call: ${name}(${JSON.stringify(args)})`)
        let result
        if (name === 'save_idea') result = saveIdea(session_id, args?.idea)
        else if (name === 'create_task') result = createTask(session_id, args?.title, args?.deadline, args?.priority)
        else if (name === 'create_plan') result = await createPlan(session_id)
        else result = { state: getState(session_id), message: 'Unknown tool.' }
        res.writeHead(200, { 'content-type': 'application/json' })
        res.end(JSON.stringify({ state: result.state, message: result.message, system_prompt: buildContextPrompt(BASE_PROMPT, result.state) }))
      } catch (error) {
        console.error('[organizer] tool processing failed:', error.message)
        res.writeHead(502, { 'content-type': 'application/json' })
        res.end(JSON.stringify({ error: 'could not process tool call' }))
      }
    })
    return
  }

  // Debug helper — no session_id lists every active session instead of
  // guessing, since querying a stale id looks exactly like broken extraction.
  if (req.url.startsWith('/api/state')) {
    const sessionId = new URL(req.url, 'http://x').searchParams.get('session_id')
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify(sessionId ? getState(sessionId) : { active_sessions: listSessionIds() }))
    return
  }

  if (req.url === '/app.js') {
    res.writeHead(200, { 'content-type': 'text/javascript' })
    res.end('(' + clientApp.toString() + ')();')
    return
  }
  res.writeHead(200, { 'content-type': 'text/html' })
  res.end(HTML)
})

// PORT when set, otherwise 3000 and up until one is free.
let port = Number(process.env.PORT) || 3000
server.on('error', (err) => {
  if (err.code === 'EADDRINUSE' && !process.env.PORT && port < 3010) {
    port += 1
    server.listen(port)
    return
  }
  throw err
})
server.on('listening', () => console.log(`Talk to it: http://localhost:${port}`))
server.listen(port)
