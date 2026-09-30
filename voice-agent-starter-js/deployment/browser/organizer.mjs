// Talk It Out — the background extraction layer.
//
// This does not touch the voice pipeline at all. It listens to finalized
// transcript turns (relayed from the browser client) and keeps a structured
// "what is this person thinking about" model per session, updated after each
// user turn via a background LLM Gateway call.
//
// State lives in memory, keyed by the Voice Agent session_id. Persistence
// across sessions is deliberately left for Module 5.

import { ApiError, llmGateway } from '../../lib.mjs'

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const sessions = new Map()

// --- module 5: persistence ---
// No accounts, no multi-user — this is a single running Thought Organizer
// that carries across separate calls, which is what "remembers past
// conversations" means for a personal assistant with one owner. A real
// multi-idea Library with tags/status filtering is future scope beyond the
// hackathon.
const DATA_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '.data')
const STATE_FILE = path.join(DATA_DIR, 'talk-it-out-state.json')

function loadPersisted() {
  try {
    return sanitize(JSON.parse(fs.readFileSync(STATE_FILE, 'utf8')))
  } catch {
    return emptyState()
  }
}

function savePersisted(state) {
  try {
    fs.mkdirSync(DATA_DIR, { recursive: true })
    fs.writeFileSync(STATE_FILE, JSON.stringify(state))
  } catch (error) {
    console.warn('[organizer] could not persist state:', error.message)
  }
}

export function resetPersisted() {
  sessions.clear()
  try {
    fs.rmSync(STATE_FILE, { force: true })
  } catch (error) {
    console.warn('[organizer] could not clear persisted state:', error.message)
  }
}

function emptyState() {
  return {
    mainGoal: '',
    tasks: [],
    ideas: [],
    goals: [],
    questions: [],
    concerns: [],
    conflicts: [],
    connections: [],
    suggestions: [],
    nextSteps: [],
  }
}

function newId(prefix) {
  return `${prefix}_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`
}

// Belt-and-suspenders: if the model forgets an id on a new item, or hands
// back something that isn't the right shape, this keeps the state usable
// instead of corrupting it.
function sanitize(state) {
  const base = emptyState()
  if (!state || typeof state !== 'object') return base
  base.mainGoal = typeof state.mainGoal === 'string' ? state.mainGoal : ''
  const strings = (arr) => (Array.isArray(arr) ? arr.filter((s) => typeof s === 'string' && s.trim()) : [])
  base.goals = strings(state.goals)
  base.questions = strings(state.questions)
  base.concerns = strings(state.concerns)
  base.conflicts = strings(state.conflicts)
  base.connections = strings(state.connections)
  base.suggestions = strings(state.suggestions)
  base.nextSteps = strings(state.nextSteps)
  base.tasks = Array.isArray(state.tasks)
    ? state.tasks
        .filter((t) => t && typeof t.title === 'string' && t.title.trim())
        .map((t) => ({
          id: typeof t.id === 'string' && t.id ? t.id : newId('t'),
          title: t.title.trim(),
          deadline: typeof t.deadline === 'string' ? t.deadline : null,
          priority: ['low', 'medium', 'high'].includes(t.priority) ? t.priority : null,
          completed: Boolean(t.completed),
        }))
    : []
  base.ideas = Array.isArray(state.ideas)
    ? state.ideas
        .filter((i) => i && typeof i.title === 'string' && i.title.trim())
        .map((i) => ({
          id: typeof i.id === 'string' && i.id ? i.id : newId('i'),
          title: i.title.trim(),
          description: typeof i.description === 'string' ? i.description : '',
          tags: Array.isArray(i.tags) ? i.tags.filter((t) => typeof t === 'string') : [],
          status: ['new', 'exploring', 'planned', 'in_progress', 'completed', 'archived'].includes(i.status)
            ? i.status
            : 'new',
        }))
    : []
  return dedupeAcrossCategories(base)
}

// Small models don't reliably follow "each thing lives in one place", so
// enforce the cheap, unambiguous cases in code: a goal that just restates a
// task or idea title is dropped, and a concern that just restates a question
// is dropped. Word-overlap only — it never rewrites anything.
const STOP = new Set(['a','an','the','to','for','of','and','or','in','on','at','my','me','i','is','it','with','up','out','about','be','do','get','go','will','should','need','want','have','has','their','your','our','that','this','so','if','into','from','by','as','are','was','can','not'])

function tokens(text) {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((w) => w && !STOP.has(w))
      .map((w) => (w.length > 4 ? w.replace(/(ing|ed|es|s)$/, '') : w))
  )
}

function isRestatement(a, b) {
  const A = tokens(a)
  const B = tokens(b)
  const smaller = Math.min(A.size, B.size)
  if (smaller < 2) return false
  let shared = 0
  for (const w of A) if (B.has(w)) shared++
  return shared / smaller >= 0.8
}

function dedupeAcrossCategories(state) {
  const titles = [...state.tasks, ...state.ideas].map((x) => x.title)
  state.goals = state.goals.filter((g) => !titles.some((t) => isRestatement(g, t)))
  state.concerns = state.concerns.filter((c) => !state.questions.some((q) => isRestatement(c, q)))
  return state
}

const SCHEMA = `{
  "mainGoal": string,
  "tasks": [{ "id": string, "title": string, "deadline": string|null, "priority": "low"|"medium"|"high"|null, "completed": boolean }],
  "ideas": [{ "id": string, "title": string, "description": string, "tags": string[], "status": "new"|"exploring"|"planned"|"in_progress"|"completed"|"archived" }],
  "goals": string[],
  "questions": string[],
  "concerns": string[],
  "conflicts": string[],
  "connections": string[],
  "suggestions": string[],
  "nextSteps": string[]
}`

const SYSTEM_PROMPT = `You maintain a structured JSON model of what a person is thinking about, built from a running spoken conversation. You do not talk to the user directly — you only output the updated JSON model.

Given the CURRENT STATE and the user's NEWEST message, return the FULL UPDATED STATE as JSON. Merge new information into what already exists rather than replacing it.

Rules:
- Never invent details that were not said or strongly implied. Never assert something about the person's schedule, availability, feelings, or decisions that they did not actually say — if they expressed uncertainty about having time, do not tell them they are free, and do not suggest a specific plan (a time estimate, a chosen option) as if they had already decided. Suggestions and next steps should help them decide, not decide for them.
- RECENT CONTEXT includes what the assistant said. The assistant's own proposals (a schedule, times, a plan, suggestions it offered) are NOT the person's decisions and are NOT things the person said. Only record a proposed time or plan as decided if the person clearly agreed to it. mainGoal, tasks and deadlines must come from the person's own words.
- When the person asks the assistant to do something ("make me a schedule", "help me plan", "organize this"), that request is not an idea and not a goal. Capture the underlying tasks they described instead, and put the plan itself in nextSteps only after they ask for it.
- If several high-priority items fall due in the same short window (for example a midterm today and assignments tomorrow), record that competition once under conflicts. Do not invent a conflict when the deadlines are spread out.
- Before finalizing, mentally check the newest message for every distinct thing the person needs to do, is thinking about, is worried about, or is asking — including small or brief commitments (a phone call, an errand, a reply they owe someone). Every one of these must show up somewhere in the state. Do not skip something just because a bigger task is already present.
- Each real-world thing the person is dealing with lives in exactly ONE place in the state. Do not restate a task as a goal, an idea as a goal, or repeat the same worry or open question in both questions and concerns. mainGoal is only for the one objective that matters most right now, not a summary of everything. When something is both a worry and a direct question, phrase it once and put it under questions if they're actually asking you something, otherwise concerns.
- mainGoal is ONE short plain sentence (under 12 words) naming the single objective that matters most right now, written the way the person would say it (e.g. "Finish the assignment by Friday"). Never a summary of several things, never third person.
- goals are only longer-term aspirations beyond the current to-do list (e.g. "Build a portfolio"). If the person didn't mention any, return an empty list — an empty goals list is correct and expected.
- ideas are only speculative things the person MIGHT do. Anything they must do, or work they already have underway, is a task, not an idea. Only set an idea's status to something other than "new" if the person said so.
- If the newest message updates, clarifies, or rephrases an existing task, idea, goal, etc., merge it into that same item — do not create a duplicate. Preserve the "id" of anything you are updating exactly as given.
- Assign a short new id (like "t3" or "i2") to genuinely new tasks or ideas only.
- Keep task and idea titles short and descriptive — a few words drawn from what the person actually said. Never use a generic placeholder title like "New idea" or "Idea" — if you can't find a specific title from what was said, use the clearest short paraphrase of the idea itself. Do not alter or substitute words in what the person actually said (e.g. do not change "newsletter" to "newspaper").
- ideas[].tags should include 1-3 short relevant keywords drawn from the idea's actual content — don't leave tags empty if there's enough detail to derive them.
- tasks[].deadline is natural text if one was mentioned ("Friday", "next week", "today"), otherwise null.
- tasks[].priority is "low" | "medium" | "high" — only set it when it's actually inferable from what was said, otherwise null.
- Add a question whenever the person expresses genuine uncertainty about what to do, asks you directly for help deciding, or asks something they don't know the answer to.
- Only add a conflict when two things genuinely compete for the same time, resource, or attention — do not manufacture conflicts.
- Suggestions must be specific to this person's actual situation. Never give generic productivity advice.
- nextSteps should name concrete next actions the person could take (e.g. "Call mom back", "Book the dentist"), phrased as options — not a decision already made on their behalf, and not a vague meta-instruction like "decide what to do."
- Only mark a task completed if the user clearly said it's done.
- Output ONLY the JSON object described below. No markdown code fences, no commentary, no explanation before or after it.

Schema:
${SCHEMA}`

// Best-effort extraction for models with no native JSON mode: strip code
// fences, then if the whole string still isn't valid JSON, scan for the
// first balanced {...} substring (handles stray preamble/trailing text that
// weaker/faster models sometimes add despite instructions not to).
function extractJson(raw) {
  const cleaned = raw.trim().replace(/^```json\s*/i, '').replace(/^```\s*/, '').replace(/```\s*$/, '')
  try {
    return JSON.parse(cleaned)
  } catch {
    // fall through to bracket matching
  }
  const start = cleaned.indexOf('{')
  if (start === -1) throw new Error('no JSON object found in model output')
  let depth = 0
  for (let i = start; i < cleaned.length; i++) {
    if (cleaned[i] === '{') depth++
    else if (cleaned[i] === '}') {
      depth--
      if (depth === 0) return JSON.parse(cleaned.slice(start, i + 1))
    }
  }
  throw new Error('no balanced JSON object found in model output')
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

// Rate limits are transient — worth a few retries with backoff before
// giving up and losing the turn's information.
async function callModel(messages, opts, attempt = 1) {
  try {
    return await llmGateway(messages, opts)
  } catch (error) {
    if (error instanceof ApiError && error.status === 429 && attempt < 4) {
      const wait = attempt * 3000 // 3s, 6s, 9s
      console.warn(`[organizer] rate limited (429), retrying in ${wait}ms (attempt ${attempt})`)
      await sleep(wait)
      return callModel(messages, opts, attempt + 1)
    }
    throw error
  }
}

async function extract(state, recentLog, newUserText) {
  const userMessage = [
    'CURRENT STATE:',
    JSON.stringify(state),
    '',
    'RECENT CONTEXT (most recent turns, oldest first):',
    recentLog.map((t) => `${t.role}: ${t.text}`).join('\n') || '(none yet)',
    '',
    'NEWEST USER MESSAGE:',
    newUserText,
    '',
    'Return the updated state as JSON.',
  ].join('\n')

  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: userMessage },
  ]
  // claude-sonnet-4-6 gives noticeably better extraction quality, but it
  // requires account-level access on LLM Gateway that isn't granted by
  // default. qwen3.5-4b-32k-fast is hosted natively by AssemblyAI and works
  // on every account. Swap via ORGANIZER_MODEL once you know what your
  // account can access.
  const opts = { model: process.env.ORGANIZER_MODEL || 'qwen3.5-4b-32k-fast', maxTokens: 1500, temperature: 0 }

  const first = await callModel(messages, opts)
  try {
    return sanitize(extractJson(first))
  } catch (error) {
    // One self-correction attempt for models without native JSON mode that
    // added stray text despite instructions.
    console.warn('[organizer] first attempt was not valid JSON, retrying once:', error.message)
    const retryMessages = [
      ...messages,
      { role: 'assistant', content: first },
      { role: 'user', content: 'That was not a valid JSON object. Reply again with ONLY the JSON object matching the schema — no words, no code fences, before or after it.' },
    ]
    try {
      const second = await callModel(retryMessages, opts)
      return sanitize(extractJson(second))
    } catch (retryError) {
      console.error('[organizer] retry also failed to parse, keeping previous state:', retryError.message)
      return state
    }
  }
}

function getSession(sessionId) {
  let session = sessions.get(sessionId)
  if (!session) {
    session = { state: loadPersisted(), log: [], pending: [], drainPromise: null }
    sessions.set(sessionId, session)
  }
  return session
}

async function drainLoop(session) {
  while (session.pending.length > 0) {
    const batch = session.pending.splice(0, session.pending.length)
    const recent = session.log.slice(-9, -1)
    session.state = await extract(session.state, recent, batch.join('\n'))
    savePersisted(session.state)
  }
}

// If a burst of turns arrives while a call is already in flight, every
// caller awaits the SAME drain loop instead of firing overlapping requests
// (which would only make a 429 situation worse). The loop keeps re-checking
// `pending`, so a turn that lands mid-drain still gets picked up and
// everyone resolves with the fully caught-up state.
function drain(session) {
  if (!session.drainPromise) {
    session.drainPromise = drainLoop(session).finally(() => {
      session.drainPromise = null
    })
  }
  return session.drainPromise
}

// role is 'user' or 'agent'. Only user turns trigger extraction — agent
// turns are kept only as context so the model can resolve things like "yeah,
// let's do that" against what it just proposed.
export async function recordTurn(sessionId, role, text) {
  const session = getSession(sessionId)
  session.log.push({ role, text })
  if (session.log.length > 20) session.log.shift() // keep the prompt small

  if (role !== 'user') return session.state

  session.pending.push(text)
  await drain(session)
  return session.state
}

// Module 3: turns the current state into plain prose the voice agent can use.
// Capped per list so the prompt stays small on a long conversation.
function summarize(state) {
  const cap = (arr) => arr.slice(0, 8)
  const lines = []
  if (state.mainGoal) lines.push(`Main goal: ${state.mainGoal}`)
  const tasks = cap(state.tasks).map(
    (t) => `${t.title}${t.completed ? ' (done)' : ''}${t.deadline ? `, due ${t.deadline}` : ''}${t.priority ? `, ${t.priority} priority` : ''}`
  )
  if (tasks.length) lines.push(`Tasks: ${tasks.join('; ')}`)
  if (state.ideas.length) lines.push(`Ideas they are considering: ${cap(state.ideas).map((i) => i.title).join('; ')}`)
  if (state.questions.length) lines.push(`Open questions: ${cap(state.questions).join('; ')}`)
  if (state.concerns.length) lines.push(`Worries: ${cap(state.concerns).join('; ')}`)
  if (state.conflicts.length) lines.push(`Competing demands: ${cap(state.conflicts).join('; ')}`)
  return lines.join('\n')
}

const CONTEXT_INTRO = `What you currently understand about this person's situation, from what they have told you so far:`
const CONTEXT_RULES = `Use this quietly. Never ask for something already listed above, and never read it back as a list. When it genuinely helps, or when they ask what you make of it, step back in one or two sentences and say what you notice: what is most urgent, what competes for their time, what is missing. Only treat something as decided if they agreed to it. Do not claim you have done anything you cannot actually do.`

// The update replaces the whole prompt, so this always starts from the base.
export function buildContextPrompt(basePrompt, state) {
  if (!basePrompt) return null
  const summary = summarize(state)
  if (!summary) return basePrompt
  return `${basePrompt}\n\n${CONTEXT_INTRO}\n${summary}\n\n${CONTEXT_RULES}`
}

// --- module 4: actions, triggered by tool calls, not by passive extraction ---
// These write to state deterministically and immediately, unlike extract()
// which is probabilistic and only runs after a full turn.

function overlap(a, b) {
  const A = tokens(a)
  const B = tokens(b)
  const smaller = Math.min(A.size, B.size)
  if (smaller === 0) return 0
  let shared = 0
  for (const w of A) if (B.has(w)) shared++
  return shared / smaller
}

function findMatch(items, ref) {
  if (!ref) return null
  let best = null
  let bestScore = 0
  for (const item of items) {
    const score = overlap(item.title, ref)
    if (score > bestScore) {
      bestScore = score
      best = item
    }
  }
  return bestScore >= 0.4 ? best : null
}

// "Saved" here means explicitly confirmed by the person this session, as
// distinct from something merely mentioned in passing (status stays "new"
// until then). Real cross-session persistence is Module 5.
export function saveIdea(sessionId, ideaRef) {
  const session = getSession(sessionId)
  const match = findMatch(session.state.ideas, ideaRef)
  if (match) {
    match.status = 'exploring'
    savePersisted(session.state)
  return { state: session.state, message: `Saved "${match.title}".` }
  }
  if (!ideaRef) return { state: session.state, message: "I didn't catch which idea to save." }
  const idea = { id: newId('i'), title: ideaRef, description: '', tags: [], status: 'exploring' }
  session.state.ideas.push(idea)
  savePersisted(session.state)
  return { state: session.state, message: `Saved "${idea.title}" as a new idea.` }
}

export function createTask(sessionId, title, deadline, priority) {
  const session = getSession(sessionId)
  if (!title) return { state: session.state, message: "I didn't catch what the task should be." }
  const match = findMatch(session.state.tasks, title)
  if (match) {
    if (deadline) match.deadline = deadline
    if (priority) match.priority = priority
    savePersisted(session.state)
  return { state: session.state, message: `Updated the task "${match.title}".` }
  }
  const task = { id: newId('t'), title, deadline: deadline || null, priority: priority || null, completed: false }
  session.state.tasks.push(task)
  savePersisted(session.state)
  return { state: session.state, message: `Added "${task.title}" as a task.` }
}

// Builds an ordered plan from what is already known — never invents new
// tasks — and writes it into nextSteps, replacing whatever was there.
export async function createPlan(sessionId) {
  const session = getSession(sessionId)
  const opts = { model: process.env.ORGANIZER_MODEL || 'qwen3.5-4b-32k-fast', maxTokens: 400, temperature: 0 }
  const messages = [
    {
      role: 'system',
      content:
        'Output ONLY a JSON object of the form {"steps": ["...", "..."]} — an ordered, concrete plan of at most 5 steps, built only from the state given below. Do not invent tasks that are not already in it. No markdown, no commentary.',
    },
    { role: 'user', content: JSON.stringify(session.state) },
  ]
  try {
    const raw = await callModel(messages, opts)
    const parsed = extractJson(raw)
    const steps = Array.isArray(parsed.steps) ? parsed.steps.filter((s) => typeof s === 'string' && s.trim()).slice(0, 5) : []
    if (steps.length) session.state.nextSteps = steps
    savePersisted(session.state)
    return {
      state: session.state,
      message: steps.length ? `Made a plan with ${steps.length} steps.` : "There wasn't quite enough here yet to build a plan.",
    }
  } catch (error) {
    console.error('[organizer] create_plan failed:', error.message)
    return { state: session.state, message: "Couldn't build a plan just now." }
  }
}

// Routed through getSession so a session id that has had no turns yet still
// reflects persisted state, not a blank slate.
export function getState(sessionId) {
  return getSession(sessionId).state
}

export function listSessionIds() {
  return [...sessions.keys()]
}
