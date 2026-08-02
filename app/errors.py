"""应用统一错误。"""


class ApiError(Exception):
    def __init__(
        self,
        message: str,
        error_type: str = 'invalid_request_error',
        status_code: int = 400,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code

    def body(self) -> dict[str, dict[str, str]]:
        return {
            'error': {
                'message': self.message,
                'type': self.error_type,
            },
        }
