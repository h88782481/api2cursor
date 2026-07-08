"""环境变量配置

使用 pydantic-settings 读取环境变量与 .env 文件。
变量名与旧版保持一致（PROXY_TARGET_URL、PROXY_API_KEY 等），部署无需改动。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # 上游中转站地址与密钥（可被管理面板配置覆盖）
    proxy_target_url: str = 'https://api.anthropic.com'
    proxy_api_key: str = ''
    # 服务监听端口
    proxy_port: int = 3029
    # 上游请求超时（秒），流式场景为块间超时
    api_timeout: int = 300
    # 访问鉴权密钥，留空不启用鉴权
    access_api_key: str = ''
    # 调试模式: off / simple / verbose
    debug_mode: str = 'off'

    def resolved_debug_mode(self) -> str:
        mode = self.debug_mode.strip().lower()
        return mode if mode in ('off', 'simple', 'verbose') else 'off'


env = EnvConfig()
