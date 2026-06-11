import subprocess
from uuid import uuid4
from Backend.app.utils import config

def clone_repo(repo_url:str)->tuple[str,str]:
    
    repo_id = str(uuid4())[:8]
    clone_path = config.CLONE_TMP / repo_id
    config.CLONE_TMP.mkdir(parents=True,exist_ok=True)

    try:
        subprocess.run(
            ["git","clone","--depth","1",repo_url,str(clone_path)],
            check=True,text=True,capture_output=True
        )

        return repo_id,str(clone_path)
    
    except subprocess.CalledProcessError as e:
        print(f"Git Clone failed:{e.stderr}")
        return "",""