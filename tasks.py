from invoke import task

import config


@task
def init(c):
    """Initialize cloned project"""
    c.run('python3 -m pip install -r requirements.txt')
