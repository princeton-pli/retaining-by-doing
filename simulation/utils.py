import inspect
import importlib


def load_module(module_path, module_name=None):
    """
    module_path: str
    """
    if module_name is None:
        module_path, module_name = module_path.rsplit('.', 1)
    
    module = importlib.import_module(module_path)
    return getattr(module, module_name)


def list_class_objects(module_path, base_class):
    """
    module_path: str
    base_class: class
    """
    module = importlib.import_module(module_path)
    # Get all classes from the settings module that inherit from BaseSetting
    setting_classes = [
        name for name, obj in inspect.getmembers(module)
        if inspect.isclass(obj) 
        and issubclass(obj, base_class) 
        and obj != base_class
    ]
        
    for setting_name in setting_classes:
        print(f"- {setting_name}")
