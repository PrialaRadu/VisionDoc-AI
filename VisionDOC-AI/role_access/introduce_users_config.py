import yaml

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def generate_hash(password):
    return pwd_context.hash(password)

def save_user_hashes(users, filepath="users.yml"):
    data = {'users': {}}

    for username, (password, role) in users.items():
        hashed = generate_hash(password)
        data['users'][username] = {
            'password': hashed,
            'role': role
        }

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

if __name__ == "__main__":
    users_to_add = {
        'admin': ('adminpwd', 'admin'),
        'tester': ('testerpwd', 'tester'),
        'user': ('userpwd', 'user'),
    }

    save_user_hashes(users_to_add)
