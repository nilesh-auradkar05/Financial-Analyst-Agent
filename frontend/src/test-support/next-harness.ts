// Test-only: lets plain `node --test` load App Router modules and run them inside a request scope.
// Import this FIRST in a test file, then load the code under test with a dynamic `await import()`:
// the resolve hook below is registered when this module runs, which is after the importing file's
// static imports were already resolved.
import "next/dist/server/node-environment-baseline.js";
import { existsSync } from "node:fs";
import * as nodeModule from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { NextRequest, NextResponse } from "next/server.js";
import { workAsyncStorage, type WorkStore } from "next/dist/server/app-render/work-async-storage.external.js";
import { workUnitAsyncStorage, type WorkUnitStore } from "next/dist/server/app-render/work-unit-async-storage.external.js";

const SRC = new URL("../", import.meta.url);

function withExtension(base: URL): URL | undefined {
  for (const suffix of [".ts", ".tsx", "/index.ts"]) {
    const candidate = new URL(base.href + suffix);
    if (existsSync(fileURLToPath(candidate))) return candidate;
  }
  return undefined;
}

type ResolveContext = { parentURL?: string };
type ResolveResult = { url: string; format?: string };
type NextResolve = (specifier: string, context: ResolveContext) => ResolveResult;
// @types/node 20 predates module.registerHooks (Node 22.15+).
const { registerHooks } = nodeModule as unknown as { registerHooks(hooks: { resolve(s: string, c: ResolveContext, next: NextResolve): ResolveResult }): void };

registerHooks({
  resolve(specifier, context, nextResolve) {
    let target: URL | undefined;
    if (specifier.startsWith("@/")) {
      const base = new URL(specifier.slice(2), SRC);
      target = /\.[cm]?[jt]sx?$/.test(base.pathname) ? base : withExtension(base);
    } else if ((specifier.startsWith("./") || specifier.startsWith("../")) && context.parentURL) {
      const base = new URL(specifier, context.parentURL);
      target = /\.[cm]?[jt]sx?$/.test(base.pathname) ? undefined : withExtension(base);
    } else if (/^next\/[^.]+$/.test(specifier)) {
      return nextResolve(`${specifier}.js`, context);
    }
    return nextResolve(target ? pathToFileURL(fileURLToPath(target)).href : specifier, context);
  },
});

/** Cookies a handler set through `cookies().set(...)` during `withRequest`, as Set-Cookie header values. */
export type RequestRun<T> = { result: T; setCookies: string[] };

/**
 * Runs `fn` as if inside a Next route handler for `req`: `cookies()` reads the request's Cookie header and
 * `cookies().set(...)` is collected into `setCookies` (a handler may equally set cookies on its Response).
 */
export async function withRequest<T>(req: Request, fn: () => Promise<T>): Promise<RequestRun<T>> {
  const incoming = new NextRequest(req.url, { headers: req.headers }).cookies;
  const sink = new NextResponse();
  const outgoing = sink.cookies;
  const jar = {
    get: (...a: Parameters<typeof incoming.get>) => incoming.get(...a),
    getAll: (...a: Parameters<typeof incoming.getAll>) => incoming.getAll(...a),
    has: (name: string) => incoming.has(name),
    set: (...a: Parameters<typeof outgoing.set>) => outgoing.set(...a),
    delete: (...a: Parameters<typeof outgoing.delete>) => outgoing.delete(...a),
  };
  const workStore = { route: new URL(req.url).pathname } as unknown as WorkStore;
  const unitStore = { type: "request", phase: "action", cookies: jar, userspaceMutableCookies: jar, mutableCookies: jar } as unknown as WorkUnitStore;
  const result = await workAsyncStorage.run(workStore, () => workUnitAsyncStorage.run(unitStore, fn));
  return { result, setCookies: sink.headers.getSetCookie() };
}
