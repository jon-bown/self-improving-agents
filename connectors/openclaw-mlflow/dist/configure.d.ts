/**
 * CLI registration for `openclaw mlflow configure` and `openclaw mlflow status`.
 */
import type { OpenClawConfig, OpenClawPluginApi } from 'openclaw/plugin-sdk/plugin-entry';
import type { mutateConfigFile } from 'openclaw/plugin-sdk/config-mutation';
type ConfigDeps = {
    loadConfig: () => OpenClawConfig;
    updateConfig: (edit: (draft: OpenClawConfig) => void) => Promise<void>;
};
type HostConfig = OpenClawPluginApi['runtime']['config'];
type ConfigMutationModule = {
    mutateConfigFile: typeof mutateConfigFile;
};
export declare function configDeps(rc: HostConfig, loadMutation?: () => Promise<ConfigMutationModule>): ConfigDeps;
export type CommanderLike = {
    command: (name: string) => CommandLike;
};
type CommandLike = {
    description: (d: string) => CommandLike;
    command: (name: string) => CommandLike;
    action: (fn: () => void | Promise<void>) => CommandLike;
};
type RegisterCliParams = {
    program: CommanderLike;
} & ConfigDeps;
declare function setPluginEntry(cfg: OpenClawConfig, config: Record<string, unknown>, enabled?: boolean): OpenClawConfig;
declare function runConfigure(deps: ConfigDeps): Promise<void>;
export { runConfigure, setPluginEntry };
export declare function registerMlflowCli(params: RegisterCliParams): void;
//# sourceMappingURL=configure.d.ts.map