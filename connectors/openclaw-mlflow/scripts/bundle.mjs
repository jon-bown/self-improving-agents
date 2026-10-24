// Bundle dist/index.js (tsc output) and its dependencies into one file, in place.
//
// OpenClaw copies a plugin's whole folder each time it loads it, several times per process. With
// @mlflow/core's dependency tree (~175 MB of node_modules) that added about two minutes to every
// `openclaw agent` run. The bundle is one file and the package declares no runtime dependencies,
// so `openclaw plugins install` has nothing to install.
import { build } from 'esbuild';

await build({
  entryPoints: ['dist/index.js'],
  outfile: 'dist/index.js',
  allowOverwrite: true,
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node22',
  // The host provides the SDK.
  external: ['openclaw', 'openclaw/*'],
  // Bundled CommonJS dependencies call require(); ESM has none.
  banner: { js: "import { createRequire as __createRequire } from 'node:module'; const require = __createRequire(import.meta.url);" },
  logLevel: 'warning',
});
