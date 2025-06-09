class TestResult:
    """Test result data class."""
    
    def __init__(self, success: bool, message: str, error_message: str = None):
        self.success = success
        self.message = message
        self.error_message = error_message 