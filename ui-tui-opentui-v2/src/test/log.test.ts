/**
 * Log hardening (boundary/log.ts):
 *   - safeStringify never throws on a circular ref / BigInt / hostile toJSON,
 *     so one bad `data` payload can't flip `fileBroken` and kill file logging.
 *   - the NDJSON file rotates by size (counter-driven), keeping disk bounded.
 * Rotation is exercised with a real temp dir; since LOG_MAX_BYTES (5 MiB) is not
 * exported, we seed the live file above the cap so the next write must rotate.
 */
import { describe, expect, test } from 'bun:test'
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { Log, safeStringify } from '../boundary/log.ts'

describe('safeStringify', () => {
  test('handles a circular object without throwing', () => {
    const a: Record<string, unknown> = { name: 'a' }
    a.self = a
    const out = safeStringify(a)
    expect(typeof out).toBe('string')
    expect(out).toContain('[Circular]')
    expect(out).toContain('"name":"a"')
  })

  test('handles a BigInt without throwing', () => {
    const out = safeStringify({ big: 10n, nested: { x: 9007199254740993n } })
    expect(typeof out).toBe('string')
    expect(out).toContain('"10n"')
    expect(out).toContain('"9007199254740993n"')
  })

  test('handles a mixed circular + BigInt payload', () => {
    const node: Record<string, unknown> = { id: 1n }
    node.parent = node
    expect(() => safeStringify({ node, list: [1n, 2n] })).not.toThrow()
  })

  test('degrades a hostile toJSON to a placeholder instead of throwing', () => {
    const hostile = {
      toJSON() {
        throw new Error('boom')
      }
    }
    let out = ''
    expect(() => {
      out = safeStringify(hostile)
    }).not.toThrow()
    expect(typeof out).toBe('string')
  })

  test('round-trips a plain object identically to JSON.stringify', () => {
    const v = { a: 1, b: 'two', c: [3, 4], d: null }
    expect(safeStringify(v)).toBe(JSON.stringify(v))
  })
})

describe('Log file logging survives a poison payload', () => {
  test('a circular/BigInt data field still writes a line and keeps filePath', () => {
    const dir = mkdtempSync(join(tmpdir(), 'hermes-log-poison-'))
    const file = join(dir, 'opentui-v2.log')
    try {
      const log = new Log(file, 'debug')
      const circular: Record<string, unknown> = {}
      circular.self = circular
      log.info('test', 'with circular', circular)
      log.info('test', 'with bigint', { n: 42n })
      // file logging must NOT be broken by the poison payloads
      expect(log.filePath).toBe(file)
      const lines = readFileLines(file)
      expect(lines.length).toBe(2)
      expect(lines[0]).toContain('[Circular]')
      expect(lines[1]).toContain('42n')
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})

describe('Log file rotation', () => {
  test('rotates the live file once it crosses the byte cap', () => {
    const dir = mkdtempSync(join(tmpdir(), 'hermes-log-rotate-'))
    const file = join(dir, 'opentui-v2.log')
    try {
      // Seed the live file ABOVE the 5 MiB cap so the very next write rotates.
      writeFileSync(file, 'x'.repeat(5 * 1024 * 1024 + 10) + '\n')
      const log = new Log(file, 'debug')
      log.info('test', 'first write after seed') // crosses the cap -> rotates
      log.info('test', 'second write on fresh file')

      const names = readdirSync(dir).sort()
      expect(names).toContain('opentui-v2.log')
      expect(names).toContain('opentui-v2.log.1') // the seeded oversized file
      // The fresh live file holds the post-rotation writes, not the seed.
      const live = readFileLines(file)
      expect(live.length).toBe(2)
      expect(live[0]).toContain('first write after seed')
      // The rotated-out file is the big seed.
      const rotated = readFileLines(`${file}.1`)
      expect(rotated[0]?.startsWith('xxxx')).toBe(true)
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})

function readFileLines(path: string): string[] {
  // trailing newline produces an empty tail we drop
  const text = readFileSync(path, 'utf8')
  return text.split('\n').filter(line => line.length > 0)
}
