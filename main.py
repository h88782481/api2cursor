"""启动入口

用法: python main.py
"""

import logging

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)

from app import create_app
from app.config import env

app = create_app()


def main():
    print(f'代理服务启动于 0.0.0.0:{env.proxy_port}')
    print(f'上游地址: {env.proxy_target_url}')
    print(f'管理面板: http://localhost:{env.proxy_port}/admin')
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=env.proxy_port,
        log_level='info',
        timeout_keep_alive=75,
        access_log=False,
    )


if __name__ == '__main__':
    main()
