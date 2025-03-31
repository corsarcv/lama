import yaml

class Meta(type):
    def __getattr__(cls, name):
        singleton = cls()
        return getattr(singleton, name)

    def __getitem__(cls, key):
            singleton = cls()
            return singleton[key]

class Config(metaclass=Meta):

    _instance = None  # Singleton
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        with open("config.yaml") as stream:
            try:
                self._config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                self._config_dict = {}
        self._initialized = True

    def __getattr__(self, name):
        if name in self._config_dict:
            return self._config_dict[name]
        else:
            raise AttributeError(f"Configuration {name} is not found")
        
    def __getitem__(self, key):
            # Allow access with square brackets
            return getattr(self, key)
