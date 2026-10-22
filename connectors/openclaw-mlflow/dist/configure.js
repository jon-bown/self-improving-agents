/**
 * CLI registration for `openclaw mlflow configure` and `openclaw mlflow status`.
 */
const PLUGIN_ID = 'mlflow-openclaw';
// OpenClaw <= 2026.9 exposes runtime.config.loadConfig/writeConfigFile. 2026.10 replaced them
// with current()/replaceConfigFile(), but its replaceConfigFile only works while the plugin's
// registration runtime is active, and the CLI retires that runtime a few seconds into a command,
// long before an interactive `configure` is answered ("runtime is no longer active"). Writes
// therefore go through the SDK's config-mutation subpath, which is not tied to the plugin runtime.
export function configDeps(rc, loadMutation = () => import('openclaw/plugin-sdk/config-mutation')) {
    const current = rc.current;
    // current() is a read-only snapshot; status only reads it.
    const loadConfig = rc.loadConfig ?? (current && (() => structuredClone(current())));
    if (!loadConfig) {
        throw new Error('mlflow-openclaw: host runtime does not expose a config API');
    }
    const legacyLoad = rc.loadConfig;
    const legacyWrite = rc.writeConfigFile;
    const updateConfig = legacyLoad && legacyWrite
        ? async (edit) => {
            const cfg = legacyLoad();
            edit(cfg);
            await legacyWrite(cfg);
        }
        : async (edit) => {
            const { mutateConfigFile } = await loadMutation();
            await mutateConfigFile({ mutate: (draft) => edit(draft), afterWrite: { mode: 'auto' } });
        };
    return { loadConfig, updateConfig };
}
function asObject(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        return {};
    }
    return value;
}
function asString(value) {
    return typeof value === 'string' ? value : '';
}
function getPluginEntry(cfg) {
    const root = asObject(cfg);
    const plugins = asObject(root.plugins);
    const entries = asObject(plugins.entries);
    const entry = asObject(entries[PLUGIN_ID]);
    const config = asObject(entry.config);
    const hooks = asObject(entry.hooks);
    return {
        enabled: typeof entry.enabled === 'boolean' ? entry.enabled : undefined,
        config,
        conversationAccess: typeof hooks.allowConversationAccess === 'boolean' ? hooks.allowConversationAccess : undefined,
    };
}
function setPluginEntry(cfg, config, enabled = true) {
    const root = asObject(cfg);
    const plugins = asObject(root.plugins);
    const entries = asObject(plugins.entries);
    const entry = asObject(entries[PLUGIN_ID]);
    // OpenClaw >= 2026.10 drops llm_input/llm_output/agent_end subscriptions from
    // non-bundled plugins unless they are granted conversation access, which would
    // leave every trace without its root and LLM spans.
    const hooks = { ...asObject(entry.hooks), allowConversationAccess: true };
    entries[PLUGIN_ID] = { ...entry, enabled, config, hooks };
    plugins.entries = entries;
    // An existing allowlist would keep the plugin from loading at all.
    if (Array.isArray(plugins.allow) && !plugins.allow.includes(PLUGIN_ID)) {
        plugins.allow = [...plugins.allow, PLUGIN_ID];
    }
    root.plugins = plugins;
    return root;
}
function showStatus(deps) {
    const cfg = deps.loadConfig();
    const entry = getPluginEntry(cfg);
    const trackingUri = asString(entry.config.trackingUri) || process.env.MLFLOW_TRACKING_URI || '(not set)';
    const experimentId = asString(entry.config.experimentId) || process.env.MLFLOW_EXPERIMENT_ID || '(not set)';
    const lines = [
        `  Enabled:        ${entry.enabled ?? 'not set'}`,
        `  Tracking URI:   ${trackingUri}`,
        `  Experiment ID:  ${experimentId}`,
        `  Conversation access: ${entry.conversationAccess ?? 'not set'}${entry.conversationAccess ? '' : ' (LLM and agent spans need it; run `openclaw mlflow configure`)'}`,
    ];
    process.stdout.write(`MLflow status:\n\n${lines.join('\n')}\n`);
}
async function runConfigure(deps) {
    // Dynamic import to avoid requiring @clack/prompts at module load time
    const p = (await import('@clack/prompts'));
    p.intro('MLflow Tracing configuration');
    const cfg = deps.loadConfig();
    const entry = getPluginEntry(cfg);
    const trackingUri = await p.text({
        message: 'MLflow Tracking URI',
        placeholder: 'http://localhost:5000',
        initialValue: asString(entry.config.trackingUri) || process.env.MLFLOW_TRACKING_URI || '',
        validate: (value) => {
            if (!value) {
                return 'Tracking URI is required';
            }
            try {
                new URL(value);
            }
            catch {
                return 'Invalid URL';
            }
        },
    });
    if (p.isCancel(trackingUri)) {
        p.cancel('Configuration cancelled');
        return;
    }
    const experimentId = await p.text({
        message: 'Experiment ID',
        placeholder: '0',
        initialValue: asString(entry.config.experimentId) || process.env.MLFLOW_EXPERIMENT_ID || '',
        validate: (value) => {
            if (!value) {
                return 'Experiment ID is required';
            }
        },
    });
    if (p.isCancel(experimentId)) {
        p.cancel('Configuration cancelled');
        return;
    }
    // Edit the file as it is now, not the snapshot read before the prompts.
    await deps.updateConfig((draft) => {
        setPluginEntry(draft, {
            ...getPluginEntry(draft).config,
            trackingUri: trackingUri,
            experimentId: experimentId,
        });
    });
    p.outro('MLflow configuration saved. Restart the gateway to apply.');
}
export { runConfigure, setPluginEntry };
export function registerMlflowCli(params) {
    const { program, loadConfig, updateConfig } = params;
    const deps = { loadConfig, updateConfig };
    const root = program.command('mlflow').description('MLflow trace export integration');
    root
        .command('configure')
        .description('Interactive setup for MLflow trace export')
        .action(() => runConfigure(deps));
    root
        .command('status')
        .description('Show current MLflow configuration')
        .action(() => {
        showStatus(deps);
    });
}
