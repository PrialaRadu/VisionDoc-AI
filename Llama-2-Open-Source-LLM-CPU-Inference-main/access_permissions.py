import box
import yaml

def authenticate(username, password):
    with open('config/users.yml', 'r', encoding='utf8') as ymlfile:
        users_cfg = box.Box(yaml.safe_load(ymlfile))
        user = users_cfg.users.get(username)
        if user and user.password == password:
            return user['role']
        return None

permissions = {
    'admin': ['search', 'view', 'describe'],
    'user': ['search', 'view']
}

def require_permission(role, action):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if action in permissions.get(role, []):
                return func(*args, **kwargs)
            else:
                raise PermissionError(f"User with role '{role}' not allowed to perform '{action}'")
        return wrapper
    return decorator





def access():
    username = input('user: ')
    password = input('password: ')

    role = authenticate(username, password)

    if not role:
        print('failed!')
        exit(1)
    return role