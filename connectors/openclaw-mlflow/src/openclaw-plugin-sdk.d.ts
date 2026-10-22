/**
 * Type declarations for the OpenClaw plugin SDK subpath imports.
 *
 * These types represent the OpenClaw plugin-sdk interface that this integration
 * targets. Once OpenClaw publishes official TypeScript types, this file can
 * be removed in favor of the `openclaw` package's own type definitions.
 */

declare module 'openclaw/plugin-sdk/plugin-entry' {
  export type OpenClawConfig = Record<string, unknown>;

  export type OpenClawPluginConfigSchema = {
    type: string;
    properties?: Record<string, unknown>;
    additionalProperties?: boolean;
  };

  export type OpenClawPluginService = {
    id: string;
    start: (ctx: {
      config: unknown;
      logger: {
        info: (message: string) => void;
        warn: (message: string) => void;
      };
    }) => void | Promise<void>;
    stop?: (ctx?: unknown) => void | Promise<void>;
  };

  export type OpenClawPluginApi = {
    pluginConfig?: unknown;
    registerService: (service: OpenClawPluginService) => void;
    registerCli: (
      register: (params: { program: unknown }) => void,
      options?: { commands?: string[] },
    ) => void;
    runtime: {
      // OpenClaw <= 2026.9 exposed loadConfig/writeConfigFile; 2026.10 replaced
      // them with current()/replaceConfigFile(). Both shapes are optional here so
      // the adapter in index.ts has to handle whichever the host provides.
      config: {
        loadConfig?: () => OpenClawConfig;
        writeConfigFile?: (cfg: OpenClawConfig) => Promise<void>;
        current?: () => OpenClawConfig;
        replaceConfigFile?: (params: {
          nextConfig: OpenClawConfig;
          afterWrite: { mode: 'auto' };
        }) => Promise<unknown>;
      };
    };
    on: (event: string, handler: (event: unknown, ctx: unknown) => void) => void;
  };

  export function definePluginEntry(options: {
    id: string;
    name: string;
    description: string;
    configSchema?: OpenClawPluginConfigSchema | (() => OpenClawPluginConfigSchema);
    register: (api: OpenClawPluginApi) => void;
  }): unknown;

  export function emptyPluginConfigSchema(): OpenClawPluginConfigSchema;
}

declare module 'openclaw/plugin-sdk/config-mutation' {
  import type { OpenClawConfig } from 'openclaw/plugin-sdk/plugin-entry';

  // OpenClaw 2026.10+: transactional write of the config file, independent of the plugin runtime.
  export function mutateConfigFile(params: {
    mutate: (draft: OpenClawConfig) => void | Promise<void>;
    afterWrite?: { mode: 'auto' } | { mode: 'restart' | 'none'; reason: string };
  }): Promise<unknown>;
}

declare module 'openclaw/plugin-sdk/diagnostic-runtime' {
  export type DiagnosticEventPayload = {
    type: string;
    sessionKey?: string;
    costUsd?: number;
    context?: {
      limit?: number;
      used?: number;
    };
    model?: string;
    provider?: string;
    durationMs?: number;
    usage?: {
      input?: number;
      output?: number;
      cacheRead?: number;
      cacheWrite?: number;
      total?: number;
    };
  };

  export function onDiagnosticEvent(handler: (event: DiagnosticEventPayload) => void): () => void;
}
