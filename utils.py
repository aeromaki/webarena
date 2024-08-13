from dotenv import dotenv_values

class DotenvException(Exception):
    pass

def load_env(key: str, env_path: str = ".env") -> str:
    if (env := dotenv_values(env_path)[key]) is None:
        raise DotenvException(f"{key} not found in .env, or .env may not exist")
    return env