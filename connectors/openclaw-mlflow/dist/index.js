import { definePluginEntry } from 'openclaw/plugin-sdk/plugin-entry';
import { createMLflowService } from './service.js';
import { configDeps, registerMlflowCli } from './configure.js';
function parsePluginConfig(raw) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
        return {};
    }
    return raw;
}
export default definePluginEntry({
    id: 'mlflow-openclaw',
    name: 'MLflow Tracing',
    description: 'Export OpenClaw LLM traces to MLflow',
    register(api) {
        const pluginConfig = parsePluginConfig(api.pluginConfig);
        const service = createMLflowService(api, pluginConfig);
        service.registerHooks();
        api.registerService(service);
        api.registerCli(({ program }) => {
            registerMlflowCli({ program: program, ...configDeps(api.runtime.config) });
        }, { commands: ['mlflow'] });
    },
});
