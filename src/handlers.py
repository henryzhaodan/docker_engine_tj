class ReadingMessageHandler:
    def __init__(self, func, config, queue):
        self.func = func
        self.config = config
        self.queue = queue

    def __call__(self, message):
        self.func(self.queue, self.config, message)


class WritingMessageHandler:
    def __init__(self, func, queue):
        self.func = func
        self.queue = queue

    def __call__(self):
        self.func(self.queue)
