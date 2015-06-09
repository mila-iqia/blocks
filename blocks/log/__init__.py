from .log import TrainingLog
from .sqlite import SQLiteLog

BACKENDS = {
    'python': TrainingLog,
    'sqlite': SQLiteLog
}
